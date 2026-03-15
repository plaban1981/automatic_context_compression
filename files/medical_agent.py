"""
Medical Research Agent using Deep Agents SDK with Autonomous Context Compression

Architecture:
- Deep Agents harness (create_deep_agent) for planning, filesystem, sub-agents
- Autonomous context compression via create_summarization_tool_middleware
- Medical-only tool suite: PubMed, ClinicalTrials.gov, FDA, WHO, OpenAlex
- Sub-agent pattern: Coordinator → Literature Agent + Trials Agent + Treatment Agent
"""

import asyncio
import os
import json
import httpx
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware


# ─────────────────────────────────────────────
#  MEDICAL SEARCH TOOLS
# ─────────────────────────────────────────────

@tool
def search_pubmed(query: str, max_results: int = 10) -> str:
    """
    Search PubMed/MEDLINE for peer-reviewed medical literature.
    Returns structured results with PMID, title, abstract, authors, journal, and year.
    
    Args:
        query: Medical search query (e.g., 'metformin type 2 diabetes treatment')
        max_results: Number of results to return (default 10, max 20)
    """
    max_results = min(max_results, 20)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    try:
        # Step 1: Search for IDs
        search_url = f"{base}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        with httpx.Client(timeout=15) as client:
            search_resp = client.get(search_url, params=search_params)
            search_data = search_resp.json()

        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results found for query: '{query}'"

        # Step 2: Fetch details
        fetch_url = f"{base}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        with httpx.Client(timeout=20) as client:
            fetch_resp = client.get(fetch_url, params=fetch_params)

        # Step 3: Parse XML
        root = ET.fromstring(fetch_resp.text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            journal_el = article.find(".//Title")
            year_el = article.find(".//PubDate/Year")
            
            authors = []
            for author in article.findall(".//Author")[:5]:
                ln = author.findtext("LastName", "")
                fn = author.findtext("ForeName", "")
                if ln:
                    authors.append(f"{ln} {fn}".strip())

            pmid = pmid_el.text if pmid_el is not None else "N/A"
            title = title_el.text if title_el is not None else "No title"
            abstract = abstract_el.text if abstract_el is not None else "No abstract available"
            journal = journal_el.text if journal_el is not None else "Unknown journal"
            year = year_el.text if year_el is not None else "Unknown year"

            results.append(
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Authors: {', '.join(authors) if authors else 'Unknown'}\n"
                f"Journal: {journal} ({year})\n"
                f"Abstract: {abstract[:500]}{'...' if abstract and len(abstract) > 500 else ''}\n"
                f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )

        return f"PubMed Results for '{query}':\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"PubMed search error: {str(e)}"


@tool
def search_clinical_trials(
    condition: str,
    intervention: str = "",
    status: str = "RECRUITING",
    max_results: int = 10
) -> str:
    """
    Search Europe PMC for clinical trial publications and registrations.
    Covers trials from PubMed, ClinicalTrials.gov, WHO ICTRP, and EU CTR.

    Args:
        condition: Medical condition (e.g., 'breast cancer', 'Alzheimer disease')
        intervention: Drug or treatment being studied (optional)
        status: Unused (kept for API compatibility)
        max_results: Number of results (default 10)
    """
    try:
        query_parts = [f'("{condition}")', 'PUB_TYPE:"Clinical Trial"']
        if intervention:
            query_parts.append(f'"{intervention}"')

        params = {
            "query": " AND ".join(query_parts),
            "resulttype": "core",
            "pageSize": min(max_results, 25),
            "format": "json",
            "cursorMark": "*",
            "sort": "CITED desc",
        }

        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params,
            )
            data = resp.json()

        items = data.get("resultList", {}).get("result", [])
        if not items:
            return f"No clinical trials found for condition: '{condition}'"

        results = []
        for r in items:
            pmid = r.get("pmid", "N/A")
            pmcid = r.get("pmcid", "")
            title = r.get("title", "No title")
            journal = r.get("journalTitle") or r.get("bookOrReportDetails", {}).get("publisher", "Unknown journal")
            year = r.get("pubYear", "N/A")
            abstract = (r.get("abstractText") or "No abstract available")[:400]
            authors = r.get("authorString", "Unknown authors")
            citation_count = r.get("citedByCount", 0)
            doi = r.get("doi", "")

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else (
                f"https://doi.org/{doi}" if doi else "N/A"
            )

            results.append(
                f"PMID: {pmid}{' | PMCID: ' + pmcid if pmcid else ''}\n"
                f"Title: {title}\n"
                f"Authors: {authors[:120]}\n"
                f"Journal: {journal} ({year}) | Citations: {citation_count}\n"
                f"Abstract: {abstract}...\n"
                f"URL: {url}\n"
            )

        return f"Clinical Trial Publications for '{condition}':\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"Europe PMC clinical trials search error: {str(e)}"


@tool
def search_fda_drugs(drug_name: str, search_type: str = "label") -> str:
    """
    Search FDA OpenFDA for drug labels, adverse events, and drug approvals.
    
    Args:
        drug_name: Name of the drug (generic or brand)
        search_type: 'label' for prescribing info, 'event' for adverse events, 'ndc' for drug lookup
    """
    try:
        endpoint_map = {
            "label": "https://api.fda.gov/drug/label.json",
            "event": "https://api.fda.gov/drug/event.json",
            "ndc": "https://api.fda.gov/drug/ndc.json",
        }
        base = endpoint_map.get(search_type, endpoint_map["label"])
        
        if search_type == "label":
            search_query = f"openfda.generic_name:{drug_name}+openfda.brand_name:{drug_name}"
        elif search_type == "event":
            search_query = f"patient.drug.medicinalproduct:{drug_name}"
        else:
            search_query = f"generic_name:{drug_name}"

        params = {
            "search": search_query,
            "limit": 5,
        }
        with httpx.Client(timeout=15) as client:
            resp = client.get(base, params=params)
            data = resp.json()

        results_raw = data.get("results", [])
        if not results_raw:
            return f"No FDA data found for drug: '{drug_name}'"

        if search_type == "label":
            output = []
            for r in results_raw[:3]:
                openfda = r.get("openfda", {})
                brand = openfda.get("brand_name", ["N/A"])[0]
                generic = openfda.get("generic_name", ["N/A"])[0]
                manufacturer = openfda.get("manufacturer_name", ["N/A"])[0]
                indications = r.get("indications_and_usage", ["N/A"])[0][:500] if r.get("indications_and_usage") else "N/A"
                warnings = r.get("warnings", ["N/A"])[0][:400] if r.get("warnings") else "N/A"
                dosage = r.get("dosage_and_administration", ["N/A"])[0][:400] if r.get("dosage_and_administration") else "N/A"

                output.append(
                    f"Brand: {brand} | Generic: {generic}\n"
                    f"Manufacturer: {manufacturer}\n"
                    f"Indications: {indications}...\n"
                    f"Warnings: {warnings}...\n"
                    f"Dosage: {dosage}...\n"
                )
            return f"FDA Drug Label Info for '{drug_name}':\n\n" + "\n---\n".join(output)

        return f"FDA data retrieved for '{drug_name}': {json.dumps(results_raw[:2], indent=2)[:1000]}"

    except Exception as e:
        return f"FDA search error: {str(e)}"


@tool
def search_medical_guidelines(topic: str, source: str = "who") -> str:
    """
    Search WHO ICD-11 / medical ontology for clinical guidelines and disease classifications.
    
    Args:
        topic: Medical topic (e.g., 'hypertension', 'diabetes mellitus type 2')
        source: 'who' for WHO ICD-11 classification, 'openalex' for research papers
    """
    try:
        if source == "who":
            # WHO ICD-11 API
            search_url = "https://id.who.int/icd/release/11/2024-01/mms/search"
            params = {
                "q": topic,
                "subtreesFilter": "",
                "includeKeywordResult": "true",
                "useFlexisearch": "false",
                "flatResults": "true",
                "highlightingEnabled": "true",
                "medicalCodingMode": "false",
            }
            headers = {
                "API-Version": "v2",
                "Accept-Language": "en",
                "Accept": "application/json",
            }
            with httpx.Client(timeout=15) as client:
                resp = client.get(search_url, params=params, headers=headers)
                data = resp.json()

            entities = data.get("destinationEntities", [])
            if not entities:
                return f"No WHO ICD-11 results for: '{topic}'"

            results = []
            for e in entities[:6]:
                code = e.get("theCode", "N/A")
                title = e.get("title", "No title")
                definition = e.get("definition", "No definition available")[:400]
                synonyms = e.get("synonyms", [])[:3]

                results.append(
                    f"ICD-11 Code: {code}\n"
                    f"Classification: {title}\n"
                    f"Definition: {definition}...\n"
                    f"Synonyms: {', '.join(synonyms) if synonyms else 'None'}\n"
                )
            return f"WHO ICD-11 Classification for '{topic}':\n\n" + "\n---\n".join(results)

        elif source == "openalex":
            # OpenAlex API for medical research works
            search_url = "https://api.openalex.org/works"
            params = {
                "search": topic,
                "filter": "concepts.id:C71924100",  # Medicine concept
                "sort": "cited_by_count:desc",
                "per_page": 8,
                "select": "id,title,publication_year,cited_by_count,primary_location,abstract_inverted_index",
            }
            headers = {"User-Agent": "MedicalResearchAgent/1.0 (mailto:research@example.com)"}
            with httpx.Client(timeout=15) as client:
                resp = client.get(search_url, params=params, headers=headers)
                data = resp.json()

            works = data.get("results", [])
            if not works:
                return f"No OpenAlex medical research found for: '{topic}'"

            results = []
            for w in works[:6]:
                title = w.get("title", "No title")
                year = w.get("publication_year", "N/A")
                citations = w.get("cited_by_count", 0)
                location = w.get("primary_location", {})
                source_name = location.get("source", {}).get("display_name", "Unknown") if location else "Unknown"
                doi = w.get("id", "").replace("https://openalex.org/", "")

                results.append(
                    f"Title: {title}\n"
                    f"Year: {year} | Citations: {citations}\n"
                    f"Journal/Source: {source_name}\n"
                    f"OpenAlex ID: {doi}\n"
                )
            return f"OpenAlex Medical Research for '{topic}':\n\n" + "\n---\n".join(results)

    except Exception as e:
        return f"Medical guidelines search error: {str(e)}"


@tool
def get_drug_interactions(drug1: str, drug2: str) -> str:
    """
    Check for known drug-drug interactions using RxNorm and DrugBank-compatible APIs.
    
    Args:
        drug1: First drug name (generic or brand)
        drug2: Second drug name (generic or brand)
    """
    try:
        # Step 1: Resolve RxCUI for each drug
        def get_rxcui(drug_name: str) -> str:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json"
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, params={"name": drug_name, "search": 1})
                data = resp.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else None

        rxcui1 = get_rxcui(drug1)
        rxcui2 = get_rxcui(drug2)

        if not rxcui1 or not rxcui2:
            missing = []
            if not rxcui1:
                missing.append(drug1)
            if not rxcui2:
                missing.append(drug2)
            return f"Could not find RxCUI for: {', '.join(missing)}. Interaction check unavailable."

        # Step 2: Check interaction
        interact_url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json"
        with httpx.Client(timeout=10) as client:
            resp = client.get(interact_url, params={"rxcuis": f"{rxcui1}+{rxcui2}"})
            data = resp.json()

        full_interactions = data.get("fullInteractionTypeGroup", [])
        if not full_interactions:
            return f"No known interactions found between {drug1} (RxCUI: {rxcui1}) and {drug2} (RxCUI: {rxcui2})."

        results = []
        for group in full_interactions:
            source = group.get("sourceName", "Unknown")
            for interaction_type in group.get("fullInteractionType", []):
                for pair in interaction_type.get("interactionPair", []):
                    severity = pair.get("severity", "Unknown")
                    description = pair.get("description", "No description")
                    results.append(
                        f"Source: {source}\n"
                        f"Severity: {severity}\n"
                        f"Interaction: {description}\n"
                    )

        header = f"Drug Interactions: {drug1} ↔ {drug2}\n"
        return header + "\n---\n".join(results[:5])

    except Exception as e:
        return f"Drug interaction check error: {str(e)}"


@tool
def summarize_research_findings(
    topic: str,
    key_papers: str,
    clinical_trials: str,
    treatment_options: str,
) -> str:
    """
    Compile and synthesize research findings into a structured medical summary.
    This tool signals a good compaction point — call COMPACT after using this tool
    to compress prior search context before generating the final report.
    
    Args:
        topic: The medical topic being researched
        key_papers: Summary of key literature findings
        clinical_trials: Summary of relevant clinical trials
        treatment_options: Summary of treatment options found
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    summary = f"""
=== MEDICAL RESEARCH SYNTHESIS ===
Topic: {topic}
Generated: {timestamp}

LITERATURE FINDINGS:
{key_papers}

CLINICAL TRIALS:
{clinical_trials}

TREATMENT OPTIONS:
{treatment_options}

EVIDENCE QUALITY NOTE:
This synthesis is based on publicly available medical databases including PubMed/MEDLINE,
ClinicalTrials.gov, FDA OpenFDA, and WHO ICD-11. All information should be verified 
by qualified healthcare professionals before clinical application.
===================================
"""
    return summary


# ─────────────────────────────────────────────
#  AGENT FACTORY
# ─────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are a specialized Medical Research Agent with deep expertise in 
clinical literature, pharmacology, and evidence-based medicine.

## YOUR MISSION
Research medical topics using ONLY peer-reviewed sources, clinical trials registries, and 
regulatory databases. You NEVER provide advice from general web searches or unverified sources.

## APPROVED MEDICAL DATABASES (use these tools only)
1. **PubMed/MEDLINE** (search_pubmed) — peer-reviewed journals, meta-analyses, RCTs
2. **ClinicalTrials.gov** (search_clinical_trials) — active and completed clinical trials  
3. **FDA OpenFDA** (search_fda_drugs) — drug labels, adverse events, approvals
4. **WHO ICD-11 / OpenAlex** (search_medical_guidelines) — clinical guidelines, disease classification
5. **RxNorm Drug Interactions** (get_drug_interactions) — evidence-based interaction checking
6. **Research Synthesizer** (summarize_research_findings) — compile findings into reports

## RESEARCH WORKFLOW
1. **Plan**: Use write_todos to outline your research steps
2. **Literature Search**: Query PubMed with precise MeSH terms when possible
3. **Clinical Evidence**: Check ClinicalTrials.gov for experimental treatments
4. **Regulatory Status**: Check FDA for approved drugs/indications
5. **Guidelines**: Check WHO ICD-11 for disease classifications
6. **Synthesize**: Use summarize_research_findings to compile — this is a COMPACTION SIGNAL
7. **Report**: Write comprehensive findings to a file

## CONTEXT COMPRESSION STRATEGY (CRITICAL)
You have access to an autonomous compact tool. Use it strategically:
- **AFTER** fetching and processing large PubMed result sets
- **BETWEEN** major research phases (literature → trials → FDA → synthesis)
- **BEFORE** writing the final comprehensive report
- **WHEN** you've extracted the key facts from a large database response

This preserves working memory for new searches while retaining extracted knowledge.

## EVIDENCE HIERARCHY (follow this order)
1. Systematic reviews & meta-analyses (highest)
2. Randomized controlled trials (RCTs)
3. Cohort studies
4. Case-control studies  
5. Expert guidelines (lowest)

## SAFETY DISCLAIMER
Always conclude reports with: "This research is for informational purposes only and should 
not substitute professional medical advice, diagnosis, or treatment."

## OUTPUT FORMAT
Structure your final reports with:
- Executive Summary
- Pathophysiology & Classification (ICD-11)
- Current Standard of Care
- Emerging Treatments & Pipeline
- Key Clinical Trials
- Drug Information (if applicable)
- Evidence Quality Assessment
- References (PMID/NCT IDs)
"""


def create_medical_research_agent(model_name: str = "anthropic:claude-sonnet-4-20250514"):
    """
    Create a Medical Research Deep Agent with autonomous context compression.
    
    The agent uses:
    - DeepAgents harness for planning, filesystem, sub-agents
    - Autonomous compaction middleware (agent decides WHEN to compress)
    - Fixed-threshold summarization as safety fallback at 85% context
    - Medical-only tool suite targeting peer-reviewed sources
    """
    model = init_chat_model(model_name)
    
    medical_tools = [
        search_pubmed,
        search_clinical_trials,
        search_fda_drugs,
        search_medical_guidelines,
        get_drug_interactions,
        summarize_research_findings,
    ]

    # Autonomous compression middleware — agent triggers this itself
    # at opportune moments (after large searches, between phases, before synthesis)
    compression_middleware = create_summarization_tool_middleware(
        model=model_name,
        backend=StateBackend,
    )

    agent = create_deep_agent(
        model=model,
        tools=medical_tools,
        system_prompt=MEDICAL_SYSTEM_PROMPT,
        middleware=[compression_middleware],
    )
    
    return agent


# ─────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────

def run_medical_research(query: str, model: str = "anthropic:claude-sonnet-4-20250514"):
    """Run a medical research query through the agent."""
    print(f"\n{'='*60}")
    print(f"🏥 Medical Research Agent")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    agent = create_medical_research_agent(model)

    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    # Extract final message
    messages = result.get("messages", [])
    if messages:
        final = messages[-1]
        content = final.content if hasattr(final, "content") else str(final)
        print("\n📋 RESEARCH FINDINGS:\n")
        print(content)
        return content
    return "No results returned."


async def run_medical_research_stream(query: str, model: str = "anthropic:claude-sonnet-4-20250514"):
    """Stream a medical research query with live output."""
    agent = create_medical_research_agent(model)
    
    print(f"\n🏥 Medical Research Agent (streaming)\n{'='*60}")
    print(f"Query: {query}\n{'='*60}\n")

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": query}]},
        version="v2",
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown_tool")
            tool_input = event.get("data", {}).get("input", {})
            print(f"\n\n🔧 [{tool_name}] → {str(tool_input)[:120]}\n", flush=True)
        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown_tool")
            if "summarize" in tool_name.lower() or "compact" in tool_name.lower():
                print(f"\n💾 Context compressed by agent at natural boundary\n", flush=True)

    print("\n\n✅ Research complete.")


if __name__ == "__main__":
    # Example queries for different use cases
    queries = {
        "basic": "What are the current treatment options for Type 2 Diabetes? Focus on recent meta-analyses and FDA-approved medications.",
        "advanced": "Research the efficacy of GLP-1 receptor agonists (semaglutide, tirzepatide) for obesity treatment. Include active clinical trials, FDA status, and drug interactions with metformin.",
        "oncology": "Summarize the latest immunotherapy advances for non-small cell lung cancer, including checkpoint inhibitors and their clinical trial data.",
    }
    
    # Run with basic query by default
    query = queries["basic"]
    
    # Check if ANTHROPIC_API_KEY is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Set it before running the agent.")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("\n📋 Agent configured successfully. Tools registered:")
        print("   ✓ search_pubmed (PubMed/MEDLINE)")
        print("   ✓ search_clinical_trials (ClinicalTrials.gov)")
        print("   ✓ search_fda_drugs (FDA OpenFDA)")
        print("   ✓ search_medical_guidelines (WHO ICD-11 + OpenAlex)")
        print("   ✓ get_drug_interactions (RxNorm)")
        print("   ✓ summarize_research_findings (Synthesis)")
        print("\n🧠 Context Compression: Autonomous (agent-controlled)")
        print("   - Agent triggers compaction at natural task boundaries")
        print("   - After extracting facts from large result sets")
        print("   - Between research phases")
        print("   - Before final report generation")
    else:
        asyncio.run(run_medical_research_stream(query))
