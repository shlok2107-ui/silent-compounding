import re

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    max_tokens=300,
)


def call_llm(prompt: str) -> str:
    return llm.invoke(prompt).content


def research_node(state):
    question = state["question"]
    context = state["context"]

    context_text = ""

    for title, sentences in zip(
        context["title"],
        context["sentences"],
    ):
        context_text += f"\nTITLE: {title}\n"

        for sentence in sentences:
            context_text += f"{sentence}\n"

    prompt = f"""
Answer the following question using ONLY the provided HotpotQA context.

Do not use outside knowledge.
Do not invent facts.
Do not claim that you searched Google, websites, databases, or other sources.
Do not mention sources that are not present in the provided context.

Question:
{question}

Provided context:
{context_text}

Identify the relevant facts from the context and use them to answer the question.

Respond in exactly this format:

FINDING: <your answer or main finding>

EVIDENCE: <the specific facts from the provided context that support the finding>

CONFIDENCE (0-1): <your confidence in the finding, as a number between 0 and 1>
"""

    output = call_llm(prompt)

    confidence = 0.0

    match = re.search(
        r"CONFIDENCE(?:\s*\(0-1\))?\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        output,
        re.IGNORECASE,
    )

    if match:
        confidence = float(match.group(1))

    structured_output = {
        "output": output,
        "confidence": confidence,
    }

    trace = state.get("trace", [])
    trace.append(
        {
            "agent": "research",
            "output": structured_output,
        }
    )

    return {
        "research_output": structured_output,
        "trace": trace,
    }