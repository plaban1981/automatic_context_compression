"""
Automatic Context Compression Pipeline
Using LangChain + Google Gemini 2.5 Pro — Native Implementation

Implements the three-layer compression strategy from Deep Agents:
  Layer 1 — Tool Result Offloading  (triggered: result > token threshold)
  Layer 2 — Tool Input Offloading   (triggered: context > 85% window)
  Layer 3 — Summarization           (triggered: context > 85% and nothing left to offload)

Plus: Autonomous Compaction Tool    (agent calls this itself at natural task boundaries)

No deepagents dependency — built entirely with:
  - langchain-google-genai
  - langchain-core
  - langgraph
"""

import os
import json
import time
import uuid
try:
    import tiktoken as _tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field

# ── LangChain / LangGraph ──────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompressionConfig:
    """Tunable knobs for the compression pipeline."""

    # Gemini 2.5 Pro has a 2M token window; we trigger at 85 %
    model_context_window: int = 1_000_000      # conservative ceiling we track against
    compression_threshold: float = 0.85        # trigger Layers 2 & 3 above this fill %
    tool_result_offload_threshold: int = 2_000  # tokens — Layer 1 threshold
    recent_messages_keep: float = 0.10          # keep last 10 % of context on summarize
    offload_dir: Path = Path("./offload_store") # filesystem for offloaded content
    max_summary_tokens: int = 800               # cap on in-context summary length
    verbose: bool = True                        # print compression events


CFG = CompressionConfig()
CFG.offload_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TOKEN COUNTING
#     Uses tiktoken (cl100k_base) when available; falls back to the standard
#     rule-of-thumb of len(text) // 4 chars-per-token for English text.
#     Gemini's tokenizer is close enough to cl100k for threshold purposes.
# ══════════════════════════════════════════════════════════════════════════════

def _load_encoder():
    try:
        import tiktoken as _tk
        return _tk.get_encoding("cl100k_base")
    except Exception:
        return None


_enc = _load_encoder()


def count_tokens(text: str) -> int:
    """Approximate token count. Accurate with tiktoken; ±10% fallback otherwise."""
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    # Fallback: ~4 chars per token is well-established for English
    return max(1, len(text) // 4)


def message_tokens(msg: BaseMessage) -> int:
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    # Add tool call overhead for AIMessage
    overhead = 0
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        overhead = sum(count_tokens(json.dumps(tc)) for tc in msg.tool_calls)
    return count_tokens(content) + overhead + 4   # 4 = per-message framing


def context_tokens(messages: list[BaseMessage]) -> int:
    return sum(message_tokens(m) for m in messages)


def context_fill(messages: list[BaseMessage]) -> float:
    return context_tokens(messages) / CFG.model_context_window


# ══════════════════════════════════════════════════════════════════════════════
# 2.  OFFLOAD STORE  (virtual filesystem for evicted content)
# ══════════════════════════════════════════════════════════════════════════════

def offload_to_disk(content: str, label: str = "content") -> str:
    """Write content to disk, return the file path."""
    fname = CFG.offload_dir / f"{label}_{uuid.uuid4().hex[:8]}.txt"
    fname.write_text(content, encoding="utf-8")
    return str(fname)


def read_offload(path: str) -> str:
    """Read previously offloaded content from disk."""
    return Path(path).read_text(encoding="utf-8")


def preview(text: str, lines: int = 10) -> str:
    """Return first N lines as a preview."""
    return "\n".join(text.splitlines()[:lines])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LAYER 1 — Tool Result Offloading
#     Fires immediately whenever a ToolMessage is too large.
# ══════════════════════════════════════════════════════════════════════════════

def layer1_offload_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Replace oversized ToolMessage bodies with a filesystem reference.
    Runs on every new message batch — O(n) scan but cheap in practice.
    """
    out = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content_str = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            tok = count_tokens(content_str)
            if tok > CFG.tool_result_offload_threshold:
                path = offload_to_disk(content_str, label=f"tool_{msg.name or 'result'}")
                preview_text = preview(content_str)
                replacement = (
                    f"[OFFLOADED — {tok:,} tokens → {path}]\n"
                    f"Preview (first 10 lines):\n{preview_text}\n"
                    f"Use read_offloaded_file('{path}') to retrieve full content."
                )
                msg = ToolMessage(
                    content=replacement,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
                if CFG.verbose:
                    print(f"  [Layer 1] Offloaded tool result ({tok:,} tok) → {path}")
        out.append(msg)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LAYER 2 — Tool Input Offloading
#     Fires when context > threshold. Evicts write/edit tool call arguments
#     from AIMessages since that content is already on disk.
# ══════════════════════════════════════════════════════════════════════════════

_WRITE_TOOLS = {"write_file", "edit_file", "create_file", "save_content"}


def layer2_offload_tool_inputs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Evict large tool-call arguments from older AIMessages.
    Targets write/edit tools whose inputs are already persisted elsewhere.
    """
    out = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            new_tool_calls = []
            for tc in msg.tool_calls:
                if tc.get("name") in _WRITE_TOOLS:
                    args = tc.get("args", {})
                    args_str = json.dumps(args)
                    if count_tokens(args_str) > 500:
                        path = offload_to_disk(args_str, label=f"input_{tc['name']}")
                        tc = {**tc, "args": {"__offloaded__": path}}
                        if CFG.verbose:
                            print(f"  [Layer 2] Evicted tool input for {tc['name']} → {path}")
                new_tool_calls.append(tc)
            msg = AIMessage(
                content=msg.content,
                tool_calls=new_tool_calls,
                id=msg.id,
            )
        out.append(msg)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LAYER 3 — Summarization
#     Fires when context > threshold AND Layers 1+2 didn't free enough space.
#     Uses Gemini to produce a structured in-context summary; original messages
#     are preserved to disk for needle-in-haystack recovery.
# ══════════════════════════════════════════════════════════════════════════════

SUMMARIZATION_PROMPT = """You are a context compression assistant.
The agent's conversation history has grown long. Your job is to produce a concise, 
structured summary that preserves everything needed to continue the task seamlessly.

Output a JSON object with exactly these keys:
{
  "session_intent": "The user's original goal in 1-2 sentences",
  "progress_so_far": "Bullet list of what has been accomplished",
  "key_facts": "Critical facts, findings, or decisions the agent must remember",
  "artifacts_created": "Files or outputs already produced (with paths/IDs)",
  "next_steps": "What the agent should do next to complete the task",
  "open_questions": "Anything unresolved or requiring user input"
}

Be dense and precise. Omit filler. The agent will continue from this summary alone.
"""


def layer3_summarize(
    messages: list[BaseMessage],
    llm: ChatGoogleGenerativeAI,
) -> list[BaseMessage]:
    """
    Summarize old messages. Keep the most recent slice (10% of window tokens).
    Returns: [SystemMessage(summary)] + recent_tail
    """
    # Determine how many recent tokens to keep
    keep_budget = int(CFG.model_context_window * CFG.recent_messages_keep)

    # Walk backwards to find the recent tail
    tail: list[BaseMessage] = []
    tail_tokens = 0
    for msg in reversed(messages):
        t = message_tokens(msg)
        if tail_tokens + t > keep_budget:
            break
        tail.insert(0, msg)
        tail_tokens += t

    # Everything before the tail goes to summary
    head = messages[: len(messages) - len(tail)]

    if not head:
        return messages   # nothing to compress

    # Persist full head to disk
    serialized = json.dumps(
        [{"role": m.__class__.__name__, "content": m.content} for m in head],
        ensure_ascii=False,
        indent=2,
    )
    archive_path = offload_to_disk(serialized, label="conversation_archive")

    # Ask Gemini to summarize
    history_text = "\n\n".join(
        f"[{m.__class__.__name__}]\n{m.content}" for m in head
        if isinstance(m.content, str)
    )
    try:
        summary_resp = llm.invoke([
            SystemMessage(content=SUMMARIZATION_PROMPT),
            HumanMessage(content=f"Summarize this conversation history:\n\n{history_text[:40_000]}")
        ])
        summary_json = summary_resp.content.strip()
        # Strip markdown fences if present
        if summary_json.startswith("```"):
            summary_json = summary_json.split("```")[1]
            if summary_json.startswith("json"):
                summary_json = summary_json[4:]
        summary_data = json.loads(summary_json)
    except Exception as e:
        # Fallback to plain text summary
        summary_data = {"session_intent": "Unknown", "progress_so_far": str(e)}

    summary_text = (
        "=== CONTEXT COMPRESSED (Layer 3 Summarization) ===\n"
        f"Archive: {archive_path}\n\n"
        + "\n".join(f"**{k.upper().replace('_',' ')}**: {v}" for k, v in summary_data.items())
        + "\n\nUse read_offloaded_file() to recover any specific detail from the archive."
    )

    compressed = [SystemMessage(content=summary_text)] + tail

    if CFG.verbose:
        saved = context_tokens(messages) - context_tokens(compressed)
        print(f"  [Layer 3] Summarized {len(head)} messages → saved ~{saved:,} tokens")
        print(f"            Archive: {archive_path}")

    return compressed


# ══════════════════════════════════════════════════════════════════════════════
# 6.  AUTONOMOUS COMPACTION TOOL
#     The agent calls this itself at natural task boundaries (from the blog post:
#     "we give the agent a compact tool it calls at opportune times")
# ══════════════════════════════════════════════════════════════════════════════

# We store the LLM reference at module level so the tool can access it
_GLOBAL_LLM: Optional[ChatGoogleGenerativeAI] = None


@tool
def compact_context(reason: str = "") -> str:
    """
    Autonomously compress the current context window.
    Call this tool at natural task boundaries, such as:
    - After completing a major research or data-gathering phase
    - Before starting a large new task where old context is irrelevant
    - After producing a deliverable and before the next distinct task
    - When you believe most prior context has become stale

    Args:
        reason: Brief explanation of why you're compacting now (for logging)

    Returns: Confirmation that compaction succeeded with token savings.
    """
    # Signal to the graph that autonomous compaction was requested
    # The actual compression happens in the graph node that reads this marker
    timestamp = datetime.now().isoformat()
    msg = f"[AUTONOMOUS COMPACT REQUESTED at {timestamp}] Reason: {reason}"
    if CFG.verbose:
        print(f"\n  [Autonomous] Agent requested compaction: {reason}")
    return msg


@tool
def read_offloaded_file(path: str) -> str:
    """
    Read content that was previously offloaded to disk during context compression.
    Use this to recover specific details that were summarized away.

    Args:
        path: The filesystem path returned by a previous offload operation
    """
    try:
        content = read_offload(path)
        if CFG.verbose:
            print(f"  [Recovery] Read offloaded file: {path} ({len(content):,} chars)")
        return content
    except FileNotFoundError:
        return f"ERROR: File not found at path '{path}'"
    except Exception as e:
        return f"ERROR reading offload: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  COMPRESSION ORCHESTRATOR
#     Decides which layer(s) to apply and in what order.
# ══════════════════════════════════════════════════════════════════════════════

class CompressionStats:
    def __init__(self):
        self.layer1_events = 0
        self.layer2_events = 0
        self.layer3_events = 0
        self.autonomous_events = 0
        self.total_tokens_saved = 0

    def report(self):
        print("\n" + "═" * 50)
        print("  COMPRESSION STATISTICS")
        print("═" * 50)
        print(f"  Layer 1 (tool result offloads):  {self.layer1_events}")
        print(f"  Layer 2 (tool input evictions):  {self.layer2_events}")
        print(f"  Layer 3 (summarizations):        {self.layer3_events}")
        print(f"  Autonomous compact calls:        {self.autonomous_events}")
        print(f"  Total tokens saved (approx):     {self.total_tokens_saved:,}")
        print("═" * 50)


STATS = CompressionStats()


def run_compression_pipeline(
    messages: list[BaseMessage],
    llm: ChatGoogleGenerativeAI,
    force_autonomous: bool = False,
) -> list[BaseMessage]:
    """
    Full three-layer compression orchestrator.

    Always runs Layer 1 (cheap, per-message).
    Runs Layers 2+3 only when context fill exceeds threshold or force_autonomous=True.
    """
    before_tokens = context_tokens(messages)

    # ── Layer 1: Always scan for oversized tool results ───────────────────
    l1_before = context_tokens(messages)
    messages = layer1_offload_tool_results(messages)
    l1_saved = l1_before - context_tokens(messages)
    if l1_saved > 0:
        STATS.layer1_events += 1
        STATS.total_tokens_saved += l1_saved

    fill = context_fill(messages)

    if fill < CFG.compression_threshold and not force_autonomous:
        return messages   # context healthy — no further compression needed

    if CFG.verbose:
        print(f"\n⚡ Compression triggered (fill={fill:.1%}, threshold={CFG.compression_threshold:.0%})")

    # ── Layer 2: Evict write-tool inputs (only if above threshold) ────────
    l2_before = context_tokens(messages)
    messages = layer2_offload_tool_inputs(messages)
    l2_saved = l2_before - context_tokens(messages)
    if l2_saved > 0:
        STATS.layer2_events += 1
        STATS.total_tokens_saved += l2_saved

    # Re-check fill after Layer 2
    fill = context_fill(messages)

    # ── Layer 3: Summarize if still above threshold ───────────────────────
    if fill >= CFG.compression_threshold or force_autonomous:
        l3_before = context_tokens(messages)
        messages = layer3_summarize(messages, llm)
        l3_saved = l3_before - context_tokens(messages)
        STATS.layer3_events += 1
        STATS.total_tokens_saved += l3_saved

    after_tokens = context_tokens(messages)
    total_saved = before_tokens - after_tokens
    if CFG.verbose and total_saved > 0:
        print(f"  Total saved this cycle: {total_saved:,} tokens | "
              f"Fill now: {context_fill(messages):.1%}")

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# 8.  LANGGRAPH AGENT  (with compression baked into the graph)
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    autonomous_compact_requested: bool


SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

## CONTEXT MANAGEMENT
You have a finite context window. To manage it wisely:

1. After completing a large data-gathering phase, call compact_context() with a reason.
2. Before starting a brand-new, unrelated task, call compact_context().  
3. After producing a deliverable and the user acknowledges it, call compact_context().
4. If you notice context is getting cluttered with stale information, call compact_context().

## TOOL RECOVERY
If a tool result was offloaded (you'll see an [OFFLOADED] notice), call 
read_offloaded_file(path) to retrieve the full content when needed.

## GENERAL BEHAVIOR
- Be thorough and precise.
- Use tools when appropriate.
- Communicate clearly about what you're doing.
"""


def build_agent(
    tools: list,
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
) -> tuple:
    """
    Build a LangGraph ReAct agent with the compression pipeline wired in.

    Returns (compiled_graph, llm)
    """
    global _GLOBAL_LLM

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0,
        max_tokens=8192,
    )
    _GLOBAL_LLM = llm

    # Always include compression tools
    all_tools = tools + [compact_context, read_offloaded_file]
    llm_with_tools = llm.bind_tools(all_tools)

    # ── Graph nodes ───────────────────────────────────────────────────────

    def call_model(state: AgentState) -> dict:
        """Run the LLM and apply Layer 1 compression to any new tool results."""
        msgs = state["messages"]

        # Apply compression pipeline before calling the model
        msgs = run_compression_pipeline(msgs, llm)

        # Prepend system prompt if not present
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

        response = llm_with_tools.invoke(msgs)
        return {
            "messages": [response],
            "autonomous_compact_requested": False,
        }

    def run_tools(state: AgentState) -> dict:
        """Execute tool calls and detect autonomous compact requests."""
        tool_node = ToolNode(all_tools)
        result = tool_node.invoke(state)

        # Check if agent called compact_context
        tool_msgs = result.get("messages", [])
        compact_requested = any(
            isinstance(m, ToolMessage) and "AUTONOMOUS COMPACT REQUESTED" in (m.content or "")
            for m in tool_msgs
        )

        if compact_requested:
            STATS.autonomous_events += 1

        return {**result, "autonomous_compact_requested": compact_requested}

    def compress_after_tools(state: AgentState) -> dict:
        """
        Post-tool-execution compression node.
        Runs Layer 1 always; Layers 2+3 if threshold exceeded or agent requested.
        """
        msgs = state["messages"]
        force = state.get("autonomous_compact_requested", False)
        msgs = run_compression_pipeline(msgs, llm, force_autonomous=force)
        return {"messages": msgs, "autonomous_compact_requested": False}

    def should_continue(state: AgentState) -> str:
        """Route: continue tool loop or end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # ── Build graph ───────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("compress", compress_after_tools)

    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "compress")
    graph.add_edge("compress", "model")

    return graph.compile(), llm


# ══════════════════════════════════════════════════════════════════════════════
# 9.  EXAMPLE TOOLS  (replace with your domain tools)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_documents(query: str, num_results: int = 5) -> str:
    """
    Search a document corpus. Returns relevant excerpts.
    (Stub — replace with your vector store / retrieval logic)
    """
    # Simulate a large result to trigger Layer 1 offloading
    fake_results = "\n\n".join([
        f"Document {i+1}: This is a detailed excerpt about '{query}'. " + ("Lorem ipsum dolor sit amet. " * 50)
        for i in range(num_results)
    ])
    return fake_results


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file. (Stub — replace with real file I/O)"""
    Path(filename).write_text(content)
    return f"Successfully wrote {len(content)} characters to {filename}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        allowed = {k: v for k, v in __import__("math").__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 10. CONVERSATION RUNNER  (with compression event reporting)
# ══════════════════════════════════════════════════════════════════════════════

def chat(
    agent,
    history: list[BaseMessage],
    user_input: str,
    verbose: bool = True,
) -> tuple[str, list[BaseMessage]]:
    """Send one turn to the agent, return (response_text, updated_history)."""
    history = history + [HumanMessage(content=user_input)]

    fill_before = context_fill(history)
    tok_before = context_tokens(history)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"User: {user_input}")
        print(f"Context: {tok_before:,} tokens ({fill_before:.1%} full)")

    result = agent.invoke({"messages": history, "autonomous_compact_requested": False})
    new_messages = result["messages"]

    # Extract final AI response
    response_text = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            response_text = msg.content
            break

    fill_after = context_fill(new_messages)
    tok_after = context_tokens(new_messages)

    if verbose:
        print(f"\nAssistant: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")
        print(f"Context: {tok_after:,} tokens ({fill_after:.1%} full)")
        if tok_before != tok_after:
            delta = tok_before - tok_after
            sign = "↓" if delta > 0 else "↑"
            print(f"Context delta: {sign}{abs(delta):,} tokens")

    return response_text, new_messages


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN — DEMO CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Automatic Context Compression Pipeline")
    print("  LangChain + Gemini 2.5 Pro")
    print("═" * 60)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠  GOOGLE_API_KEY not set. Set it to run the live agent.")
        print("   export GOOGLE_API_KEY='your-key-here'")
        print("\n✅ Pipeline configured. Summary:")
        print(f"   Model context window:         {CFG.model_context_window:,} tokens")
        print(f"   Compression threshold:        {CFG.compression_threshold:.0%}")
        print(f"   Layer 1 offload threshold:    {CFG.tool_result_offload_threshold:,} tokens")
        print(f"   Recent messages kept:         {CFG.recent_messages_keep:.0%} of window")
        print(f"   Offload directory:            {CFG.offload_dir}")
        print("\n  Compression Layers:")
        print("   Layer 1 — Tool Result Offloading  (always-on, per-message)")
        print("   Layer 2 — Tool Input Eviction     (fires at 85% fill)")
        print("   Layer 3 — LLM Summarization       (fires when L2 insufficient)")
        print("   Bonus   — Autonomous Compact Tool  (agent calls this itself)")
        return

    # Build agent
    user_tools = [search_documents, write_file, calculate]
    agent, llm = build_agent(
        tools=user_tools,
        model_name="gemini-2.5-pro",
        api_key=api_key,
    )

    history: list[BaseMessage] = []

    # Demo multi-turn conversation
    demo_turns = [
        "Search for information about climate change impacts on agriculture. Get 8 results.",
        "Now search for water conservation techniques used in sustainable farming. Get 5 results.",
        "Summarize the key findings from both searches into a report and save it to climate_report.txt.",
        "What's 1234 * 5678 + 9012?",
        "Now I want to start a completely new task: search for information about renewable energy policy.",
    ]

    for turn in demo_turns:
        response, history = chat(agent, history, turn)

    STATS.report()


if __name__ == "__main__":
    main()
