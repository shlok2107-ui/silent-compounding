import re

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0.5, max_tokens=500)


def call_llm(prompt: str) -> str:
    return llm.invoke(prompt).content


def action_node(state):
    question = state["question"]
    summary = state["summary_output"]

    prompt = f"""
Answer the user's question using the provided summary.

Question:
{question}

Summary:
{summary}

Give a clear final answer based only on the information in the summary.

Respond in exactly this format:

FINAL ANSWER: <your answer to the question>

CONFIDENCE (0-1): <your confidence in the final answer, as a number between 0 and 1>
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
        "agent": "action",
        "output": structured_output
    })

    return {
        "action_output": structured_output,
        "trace": trace
    }