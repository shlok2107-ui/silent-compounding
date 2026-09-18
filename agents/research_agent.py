import re

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0.5, max_tokens=500)


def call_llm(prompt: str) -> str:
    return llm.invoke(prompt).content


def research_node(state):
    question = state["question"]

    prompt = f"""
Research this question thoroughly:

{question}

Respond in exactly this format:

FINDING: <your answer or main finding>

EVIDENCE: <the key facts or information that support the finding>

CONFIDENCE (0-1): <your confidence in the finding, as a number between 0 and 1>
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
        "agent": "research",
        "output": structured_output
    })

    return {
        "research_output": structured_output,
        "trace": trace
    }