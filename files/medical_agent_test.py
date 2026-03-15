from medical_agent import create_medical_research_agent
import os
from dotenv import load_dotenv
load_dotenv()

agent = create_medical_research_agent("anthropic:claude-sonnet-4-20250514")

if __name__ == "__main__":
    print("Asking the medical research agent about Type 2 Diabetes treatment options...")
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "What are the current treatment options for Type 2 Diabetes? "
                    "Include recent meta-analyses, FDA-approved GLP-1 agonists, "
                    "and active clinical trials."
        }]
    })
    print(result["messages"][-1].content)