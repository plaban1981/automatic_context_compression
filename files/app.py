"""
Streamlit UI for Medical Research Agent
A production-grade interface for the Deep Agents medical research system.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from queue import Queue, Empty
from threading import Thread

import streamlit as st
from streamlit.components.v1 import html

# Page configuration
st.set_page_config(
    page_title="MedResearch AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --medical-deep: #0a1628;
    --medical-blue: #1e3a5f;
    --medical-teal: #0d9488;
    --medical-cyan: #22d3ee;
    --medical-green: #10b981;
    --medical-amber: #f59e0b;
    --medical-red: #ef4444;
    --surface: #0f2040;
    --surface-2: #162b4a;
    --border: rgba(34,211,238,0.15);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --font-display: 'DM Serif Display', serif;
    --font-mono: 'JetBrains Mono', monospace;
    --font-body: 'Inter', sans-serif;
}

* { font-family: var(--font-body); }

.stApp {
    background: linear-gradient(135deg, var(--medical-deep) 0%, #071020 100%);
}

/* Header */
.med-header {
    background: linear-gradient(90deg, var(--medical-blue), #0d2240);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.med-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(13,148,136,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.med-header h1 {
    font-family: var(--font-display);
    font-size: 2.6rem;
    color: #f8fafc;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.med-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.med-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(13,148,136,0.15);
    border: 1px solid rgba(13,148,136,0.35);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    color: var(--medical-teal);
    margin-top: 12px;
}

/* Cards */
.tool-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.tool-card:hover { border-color: rgba(34,211,238,0.35); }
.tool-card .tool-icon { font-size: 1.4rem; margin-bottom: 6px; }
.tool-card .tool-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text-primary);
}
.tool-card .tool-desc {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 3px;
    line-height: 1.4;
}

/* Activity feed */
.activity-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    background: var(--surface);
    border-left: 3px solid transparent;
    border-radius: 0 8px 8px 0;
    margin-bottom: 8px;
    font-size: 0.82rem;
    animation: fadeIn 0.3s ease;
}
.activity-item.tool { border-color: var(--medical-cyan); }
.activity-item.compact { border-color: var(--medical-amber); }
.activity-item.model { border-color: var(--medical-teal); }
.activity-item.error { border-color: var(--medical-red); }
.activity-time {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-secondary);
    white-space: nowrap;
    margin-top: 2px;
}

/* Result container */
.result-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
    white-space: pre-wrap;
    font-family: var(--font-body);
    max-height: 600px;
    overflow-y: auto;
}

/* Context compression indicator */
.compression-widget {
    background: linear-gradient(90deg, rgba(245,158,11,0.08), rgba(16,185,129,0.08));
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}
.compression-bar {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
}
.compression-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--medical-amber), var(--medical-teal));
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* Query templates */
.query-pill {
    display: inline-block;
    background: rgba(30,58,95,0.6);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--medical-cyan);
    cursor: pointer;
    transition: all 0.2s;
    margin: 4px;
}
.query-pill:hover {
    background: rgba(13,148,136,0.2);
    border-color: var(--medical-teal);
}

/* Metrics row */
.metric-chip {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
}
.metric-chip .val {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--medical-cyan);
}
.metric-chip .lbl {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    border-radius: 10px !important;
}
.stButton button {
    background: linear-gradient(135deg, var(--medical-teal), #0a7a70) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.9 !important; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-8px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.pulse { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "api_key": "",
    "activity_log": [],
    "result": "",
    "running": False,
    "tool_calls": 0,
    "compression_count": 0,
    "sources_found": 0,
    "start_time": None,
    "elapsed": 0,
    "queue": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="med-header">
    <h1>🔬 MedResearch AI</h1>
    <p>Evidence-based medical research powered by Deep Agents + Autonomous Context Compression</p>
    <div class="med-badge">⚡ Deep Agents SDK &nbsp;|&nbsp; 🧠 Auto-Compact &nbsp;|&nbsp; 📚 PubMed · ClinicalTrials · FDA · WHO</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-ant-...",
        help="Your Anthropic API key for Claude",
    )
    if api_key:
        st.session_state.api_key = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key

    model_choice = st.selectbox(
        "Model",
        ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
        index=0,
        help="Sonnet recommended for research quality vs. speed tradeoff",
    )

    st.markdown("---")
    st.markdown("### 🛠️ Medical Tool Suite")
    
    tools_info = [
        ("📖", "PubMed/MEDLINE", "Peer-reviewed literature, meta-analyses, RCTs"),
        ("🧪", "Europe PMC Trials", "Clinical trial publications via Europe PMC"),
        ("💊", "FDA OpenFDA", "Drug labels, adverse events, approvals"),
        ("🌍", "WHO ICD-11", "Disease classifications & guidelines"),
        ("🔗", "OpenAlex", "High-citation medical research"),
        ("⚗️", "RxNorm", "Drug-drug interaction checking"),
    ]
    for icon, name, desc in tools_info:
        st.markdown(f"""
        <div class="tool-card">
            <div class="tool-icon">{icon}</div>
            <div class="tool-name">{name}</div>
            <div class="tool-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Context Compression")
    st.markdown("""
    <div style="font-size:0.8rem; color:#94a3b8; line-height:1.6">
    The agent autonomously triggers context compression at:
    <ul style="margin-top:8px; padding-left:1.2rem;">
    <li>Task phase boundaries</li>
    <li>After large result sets</li>
    <li>Before synthesis steps</li>
    <li>Before final report</li>
    </ul>
    Falls back to auto-compact at 85% context limit.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  QUERY TEMPLATES
# ─────────────────────────────────────────────
st.markdown("**Quick Research Templates:**")

templates = [
    "Type 2 Diabetes — GLP-1 agonist efficacy vs metformin",
    "Non-small cell lung cancer — immunotherapy checkpoint inhibitors",
    "Alzheimer's disease — amyloid-targeting treatments & trials",
    "Hypertension — ACE inhibitors vs ARBs in heart failure",
    "Drug interaction: semaglutide + metformin safety profile",
    "COVID-19 long-haul — current treatment protocols & evidence",
]

template_html = "".join(
    f'<span class="query-pill" onclick="document.querySelector(\'textarea\').value=\'{t}\'">{t}</span>'
    for t in templates
)
st.markdown(f'<div style="margin-bottom:20px">{template_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN INPUT
# ─────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_area(
        "Research Query",
        placeholder="e.g., What are the current treatment options for Type 2 Diabetes? Include recent meta-analyses, FDA-approved medications, and active clinical trials.",
        height=100,
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Research", use_container_width=True, disabled=st.session_state.running)


# ─────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-chip">
        <div class="val">{st.session_state.tool_calls}</div>
        <div class="lbl">Tool Calls</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-chip">
        <div class="val">{st.session_state.compression_count}</div>
        <div class="lbl">Auto Compressions</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-chip">
        <div class="val">{st.session_state.sources_found}</div>
        <div class="lbl">Sources Found</div>
    </div>""", unsafe_allow_html=True)
with m4:
    elapsed = st.session_state.elapsed
    st.markdown(f"""<div class="metric-chip">
        <div class="val">{elapsed:.0f}s</div>
        <div class="lbl">Research Time</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ACTIVITY + RESULTS LAYOUT
# ─────────────────────────────────────────────
col_activity, col_results = st.columns([2, 3])

with col_activity:
    st.markdown("#### 📡 Agent Activity")
    activity_placeholder = st.empty()

with col_results:
    st.markdown("#### 📋 Research Findings")
    results_placeholder = st.empty()


def render_activity(log):
    if not log:
        activity_placeholder.markdown(
            '<div style="color:#475569;font-size:0.82rem;padding:16px">Waiting for research query...</div>',
            unsafe_allow_html=True
        )
        return
    items_html = ""
    for item in log[-20:]:  # show last 20 items
        cls = item.get("type", "model")
        icon = {"tool": "🔧", "compact": "🧠", "model": "💬", "error": "❌"}.get(cls, "•")
        items_html += f"""
        <div class="activity-item {cls}">
            <span>{icon}</span>
            <div>
                <div style="color:#e2e8f0">{item['msg'][:80]}</div>
                <div class="activity-time">{item['time']}</div>
            </div>
        </div>"""
    activity_placeholder.markdown(items_html, unsafe_allow_html=True)


def render_result(text):
    if text:
        results_placeholder.markdown(
            f'<div class="result-box">{text}</div>',
            unsafe_allow_html=True
        )
    else:
        results_placeholder.markdown(
            '<div style="color:#475569;font-size:0.85rem;padding:24px">Results will appear here after research completes.</div>',
            unsafe_allow_html=True
        )


# Initial renders
render_activity(st.session_state.activity_log)
render_result(st.session_state.result)


# ─────────────────────────────────────────────
#  AGENT RUNNER
# ─────────────────────────────────────────────
def add_activity(msg: str, kind: str = "model"):
    st.session_state.activity_log.append({
        "msg": msg,
        "type": kind,
        "time": datetime.now().strftime("%H:%M:%S"),
    })


def run_agent_thread(query: str, model_name: str, result_queue: Queue):
    """Run agent in a background thread, pushing updates to a queue."""
    try:
        # Import here to avoid import errors if deps missing
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from medical_agent import create_medical_research_agent
        
        agent = create_medical_research_agent(f"anthropic:{model_name}")
        
        result_text = ""
        tool_count = 0
        compress_count = 0
        sources = 0

        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content") and last.content:
                    if hasattr(last, "type") and last.type == "ai":
                        result_text = last.content if isinstance(last.content, str) else str(last.content)
                        result_queue.put(("result", result_text))
                    elif hasattr(last, "name"):  # tool message
                        tool_name = getattr(last, "name", None) or "tool"
                        content = str(last.content)[:100] if last.content else ""
                        tool_count += 1
                        # Count sources
                        if "pubmed" in tool_name.lower():
                            sources += content.lower().count("pmid:")
                        elif "clinical" in tool_name.lower():
                            sources += content.lower().count("nct id:")
                        
                        if "compact" in tool_name.lower() or "summariz" in tool_name.lower():
                            compress_count += 1
                            result_queue.put(("activity", ("🧠 Context compressed autonomously", "compact")))
                        else:
                            result_queue.put(("activity", (f"Tool: {tool_name}", "tool")))
                        
                        result_queue.put(("metrics", (tool_count, compress_count, sources)))

        result_queue.put(("done", result_text))

    except ImportError as e:
        result_queue.put(("error", f"Import error: {e}. Ensure deepagents is installed: pip install deepagents"))
    except Exception as e:
        result_queue.put(("error", str(e)))


# ─────────────────────────────────────────────
#  HANDLE RUN
# ─────────────────────────────────────────────
if run_btn and query and not st.session_state.running:
    if not st.session_state.api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
    else:
        st.session_state.running = True
        st.session_state.activity_log = []
        st.session_state.result = ""
        st.session_state.tool_calls = 0
        st.session_state.compression_count = 0
        st.session_state.sources_found = 0
        st.session_state.start_time = time.time()

        add_activity(f"Starting research: {query[:60]}...", "model")
        add_activity("Initializing Deep Agents + compression middleware", "model")

        q = Queue()
        st.session_state.queue = q
        thread = Thread(target=run_agent_thread, args=(query, model_choice, q), daemon=True)
        thread.start()
        st.rerun()

# Poll queue on every rerun while agent is running
if st.session_state.running and st.session_state.queue is not None:
    q = st.session_state.queue
    done = False
    while True:
        try:
            msg_type, payload = q.get_nowait()
            if msg_type == "activity":
                text, kind = payload
                add_activity(text, kind)
            elif msg_type == "result":
                st.session_state.result = payload
            elif msg_type == "metrics":
                tc, cc, sc = payload
                st.session_state.tool_calls = tc
                st.session_state.compression_count = cc
                st.session_state.sources_found = sc
            elif msg_type == "done":
                st.session_state.result = payload
                st.session_state.running = False
                st.session_state.elapsed = time.time() - st.session_state.start_time
                st.session_state.queue = None
                add_activity("✅ Research complete!", "model")
                done = True
                break
            elif msg_type == "error":
                st.error(f"Agent error: {payload}")
                st.session_state.running = False
                st.session_state.queue = None
                done = True
                break
        except Empty:
            break

    render_activity(st.session_state.activity_log)
    render_result(st.session_state.result)

    if st.session_state.running:
        if st.session_state.start_time:
            st.session_state.elapsed = time.time() - st.session_state.start_time
        time.sleep(0.5)
        st.rerun()


# ─────────────────────────────────────────────
#  ARCHITECTURE INFO
# ─────────────────────────────────────────────
with st.expander("🏗️ Architecture: How this agent works", expanded=False):
    st.markdown("""
    ```
    User Query
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │  create_deep_agent (LangGraph compiled graph)       │
    │                                                     │
    │  ┌─────────────────┐   ┌──────────────────────┐    │
    │  │  Planning Tool  │   │  Summarization Tool  │    │
    │  │  (write_todos)  │   │  Middleware (AUTO-   │    │
    │  │                 │   │  COMPACT at agent's  │    │
    │  │  read_todos     │   │  discretion)         │    │
    │  └─────────────────┘   └──────────────────────┘    │
    │                                                     │
    │  Medical Tool Suite (6 tools)                       │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │ PubMed   │ │ Clinical │ │  FDA     │           │
    │  │ MEDLINE  │ │ Trials   │ │ OpenFDA  │           │
    │  └──────────┘ └──────────┘ └──────────┘           │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │ WHO      │ │ RxNorm   │ │ Research │           │
    │  │ ICD-11   │ │ Drug Int │ │ Synth.   │           │
    │  └──────────┘ └──────────┘ └──────────┘           │
    │                                                     │
    │  Context Compression Strategy:                      │
    │  • Agent triggers compact at natural boundaries     │
    │  • Auto-fallback at 85% context window fill         │
    │  • Preserves 10% recent messages on compaction      │
    │  • All history saved to virtual filesystem          │
    └─────────────────────────────────────────────────────┘
        │
        ▼
    Structured Medical Research Report
    ```
    
    **Key concepts from the blog post applied:**
    - Agent decides **when** to compact (not at fixed thresholds)
    - Compaction triggered after extracting facts from large PubMed results
    - Compaction triggered between major research phases  
    - Conservative compaction — agent preserves relevant context
    - `create_summarization_tool_middleware` exposes compact as an agent tool
    """)

st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.72rem; margin-top:24px">
⚕️ This tool is for research purposes only. Not a substitute for professional medical advice.
&nbsp;|&nbsp; Powered by Deep Agents SDK + LangGraph + LangChain
</div>
""", unsafe_allow_html=True)
