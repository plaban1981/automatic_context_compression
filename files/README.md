# 🔬 Medical Research Agent

A specialized AI research agent powered by **LangChain Deep Agents SDK** with **Autonomous Context Compression**, designed exclusively for peer-reviewed medical databases and clinical data sources.

---

## Architecture

```
User Query
    │
    ▼
create_deep_agent (LangGraph compiled graph)
    │
    ├── Planning Tools (write_todos / read_todos)
    │
    ├── Summarization Tool Middleware ← AUTONOMOUS CONTEXT COMPRESSION
    │   └── Agent decides WHEN to compact (not fixed threshold)
    │       ├── After large PubMed result sets
    │       ├── Between research phases
    │       ├── Before synthesis / final report
    │       └── Fallback: auto-compact at 85% context window
    │
    └── Medical Tool Suite
        ├── search_pubmed         → PubMed/MEDLINE (peer-reviewed literature)
        ├── search_clinical_trials → ClinicalTrials.gov (active trials)
        ├── search_fda_drugs      → FDA OpenFDA (approvals, labels, adverse events)
        ├── search_medical_guidelines → WHO ICD-11 + OpenAlex
        ├── get_drug_interactions → RxNorm (evidence-based interactions)
        └── summarize_research_findings → Synthesis (signals compact point)
```

## Autonomous Context Compression

Based on the [LangChain blog post](https://blog.langchain.com/autonomous-context-compression/), this agent uses `create_summarization_tool_middleware` which:

1. **Exposes a compact tool to the agent** — the model decides when to call it
2. **Agent compacts at opportune moments**, not at fixed token thresholds:
   - After extracting facts from a large set of PubMed results
   - At clean task boundaries (e.g., done with literature → starting trials phase)
   - Before generating a long synthesis report
3. **Retains 10% recent messages** on each compaction
4. **Preserves full history** in the virtual filesystem for recovery
5. **Falls back** to automatic compaction at 85% context limit

This is better than fixed-threshold compaction because:
- Compacting mid-literature-review would lose relevant context
- Compacting after extracting key facts is the ideal moment
- The agent's reasoning model understands task structure better than a fixed counter

## Installation

```bash
pip install -r requirements.txt
```

Or with uv:
```bash
uv add deepagents langchain langchain-anthropic httpx streamlit
```

## Usage

### Run the Streamlit UI
```bash
export ANTHROPIC_API_KEY="your-key-here"
streamlit run app.py
```

### Use the Agent Programmatically
```python
from medical_agent import create_medical_research_agent

agent = create_medical_research_agent("anthropic:claude-sonnet-4-20250514")

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What are the current treatment options for Type 2 Diabetes? "
                   "Include recent meta-analyses, FDA-approved GLP-1 agonists, "
                   "and active clinical trials."
    }]
})
print(result["messages"][-1].content)
```

### Streaming with Activity Events
```python
import asyncio
from medical_agent import run_medical_research_stream

asyncio.run(run_medical_research_stream(
    "Efficacy of immunotherapy in non-small cell lung cancer"
))
```

## Medical Data Sources

| Tool | Source | Data Type |
|------|--------|-----------|
| `search_pubmed` | NCBI E-utilities | Peer-reviewed literature, abstracts, PMIDs |
| `search_clinical_trials` | ClinicalTrials.gov API v2 | Trial phases, status, eligibility |
| `search_fda_drugs` | FDA OpenFDA API | Labels, adverse events, manufacturer info |
| `search_medical_guidelines` | WHO ICD-11 + OpenAlex | Disease classification, citation-ranked research |
| `get_drug_interactions` | RxNav/RxNorm NLM API | Evidence-based DDI severity & description |
| `summarize_research_findings` | Internal synthesis | Structured report compilation |

## Research Workflow

1. **Plan**: Agent uses `write_todos` to outline phases
2. **Literature**: PubMed search with MeSH-aligned terms
3. **Compress** (autonomous) — agent compacts after extracting key papers
4. **Clinical Trials**: ClinicalTrials.gov for experimental evidence
5. **Regulatory**: FDA for approved status and labeling
6. **Guidelines**: WHO ICD-11 classification
7. **Compress** (autonomous) — before synthesis
8. **Synthesize**: Structured report with evidence quality assessment
9. **Save**: Agent writes full report to filesystem

## Evidence Hierarchy

1. Systematic reviews & meta-analyses (highest quality)
2. Randomized controlled trials (RCTs)
3. Cohort studies
4. Case-control studies
5. Expert guidelines (lowest)

## Sample Queries

- `"What are the current treatment options for Type 2 Diabetes? Include recent meta-analyses and FDA-approved medications."`
- `"Research the efficacy of GLP-1 receptor agonists for obesity. Include active clinical trials and drug interactions with metformin."`
- `"Summarize immunotherapy advances for non-small cell lung cancer, including checkpoint inhibitor trial data."`
- `"What are the latest Alzheimer's disease treatments? Focus on amyloid-targeting therapies and their clinical trial status."`

## Disclaimer

This agent is for research and informational purposes only. All findings should be verified by qualified healthcare professionals before any clinical application.
