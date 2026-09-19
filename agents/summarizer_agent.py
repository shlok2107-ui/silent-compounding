import re

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0.5, max_tokens=250)


def call_llm(prompt: str) -> str:
    return llm.invoke(prompt).content


def summarize_node(state):
    research_output = state["research_output"]

    prompt = f"""
Summarize the following research output.

Preserve:
- The main finding
- The important supporting facts
- The stated confidence

Do not add new facts or information that is not present in the research output.

Research output:

{research_output}

Respond in exactly this format:

SUMMARY: <concise summary of the main finding and supporting facts>

CONFIDENCE (0-1): <the confidence stated in the research output>
"""

    output = call_llm(prompt)

    confidence = 0.0

    match = re.search(
        r"CONFIDENCE(?:\s*\(0-1\))?\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        output,
        re.IGNORECASE
    )

    if match:
        confidence = float(match.group(1))

    structured_output = {
        "output": output,
        "confidence": confidence
    }

    trace = state.get("trace", [])
    trace.append({
        "agent": "summarizer",
        "output": structured_output
    })

    return {
        "summary_output": structured_output,
        "trace": trace
    }