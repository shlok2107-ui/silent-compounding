import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from langchain_ollama import ChatOllama


@dataclass
class ErrorTemplate:
    error_type: str
    original_claim_pattern: str
    injected_claim: str
    description: str


ERROR_BANK = {
    "factual": [],
    "numerical": [],
    "false_premise": [],
}


LLM = ChatOllama(
    model="llama3.1",
    temperature=0,
    num_predict=250,
)


def _extract_numbers(text: str) -> list[str]:
    return re.findall(
        r"\b\d[\d,]*(?:\.\d+)?\b",
        text,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(
        r"\w+|[^\w\s]",
        text,
    )


def _extract_finding(research_output: str) -> str:
    match = re.search(
        r"FINDING:\s*(.*?)\s*"
        r"(?=EVIDENCE:|CONFIDENCE(?:\s*\([^)]*\))?\s*:|$)",
        research_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Could not extract FINDING from Research Agent output."
        )

    finding = match.group(1).strip()

    if not finding:
        raise ValueError(
            "Research Agent FINDING is empty."
        )

    return finding


def _extract_candidate_claims(research_output: str) -> list[str]:
    finding = _extract_finding(research_output)

    candidates = [finding]

    evidence_match = re.search(
        r"EVIDENCE:\s*(.*?)\s*"
        r"(?=CONFIDENCE(?:\s*\([^)]*\))?\s*:|$)",
        research_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if evidence_match:
        evidence = evidence_match.group(1).strip()

        sentences = re.split(
            r"(?<=[.!?])\s+",
            evidence,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                candidates.append(sentence)

    return candidates


def _validate_factual_substitution(
    original_claim: str,
    corrupted_claim: str,
) -> None:
    original_tokens = _tokenize(original_claim)
    corrupted_tokens = _tokenize(corrupted_claim)

    matcher = SequenceMatcher(
        None,
        original_tokens,
        corrupted_tokens,
    )

    opcodes = matcher.get_opcodes()

    changed = [
        opcode
        for opcode in opcodes
        if opcode[0] != "equal"
    ]

    if len(changed) != 1:
        raise ValueError(
            "Factual injection must contain exactly one contiguous "
            "changed span."
        )

    tag, i1, i2, j1, j2 = changed[0]

    if tag != "replace":
        raise ValueError(
            "Factual injection must replace existing text; "
            "insertions and deletions are not allowed."
        )

    if i1 == i2 or j1 == j2:
        raise ValueError(
            "Factual injection must replace existing text "
            "with replacement text."
        )

    original_numbers = _extract_numbers(original_claim)
    corrupted_numbers = _extract_numbers(corrupted_claim)

    if original_numbers != corrupted_numbers:
        raise ValueError(
            "Factual injection changed a numerical value. "
            "Use numerical error type for number changes."
        )


def _validate_numerical_corruption(
    original_claim: str,
    corrupted_claim: str,
) -> None:
    original_numbers = _extract_numbers(original_claim)
    corrupted_numbers = _extract_numbers(corrupted_claim)

    if len(original_numbers) == 0:
        raise ValueError(
            "Numerical injection requires an existing numerical value."
        )

    if len(original_numbers) != len(corrupted_numbers):
        raise ValueError(
            "Numerical injection must preserve the number of "
            "numerical values."
        )

    differences = [
        (old, new)
        for old, new in zip(
            original_numbers,
            corrupted_numbers,
        )
        if old != new
    ]

    if len(differences) != 1:
        raise ValueError(
            "Numerical injection must change exactly one number."
        )


def _validate_false_premise_corruption(
    original_claim: str,
    corrupted_claim: str,
) -> None:
    if original_claim == corrupted_claim:
        raise ValueError(
            "False-premise injection did not change the claim."
        )

    if len(corrupted_claim) <= len(original_claim):
        raise ValueError(
            "False-premise injection must add a false premise "
            "to the original claim."
        )


def _validate_corruption(
    original_claim: str,
    corrupted_claim: str,
    error_type: str,
) -> None:
    if not corrupted_claim:
        raise ValueError(
            "Generated corrupted claim is empty."
        )

    if error_type == "factual":
        _validate_factual_substitution(
            original_claim,
            corrupted_claim,
        )

    elif error_type == "numerical":
        _validate_numerical_corruption(
            original_claim,
            corrupted_claim,
        )

    elif error_type == "false_premise":
        _validate_false_premise_corruption(
            original_claim,
            corrupted_claim,
        )

    else:
        raise ValueError(
            f"Unsupported error type: {error_type}"
        )


def _generate_factual_corruption(
    original_claim: str,
) -> ErrorTemplate:
    """
    Ask the LLM to identify one factual element to replace.

    IMPORTANT:
    The LLM does NOT rewrite the complete claim.

    Python performs the actual replacement so that the corruption
    is guaranteed to be a single controlled substitution.
    """

    prompt = f"""
You are creating a controlled factual error for an experiment.

Original factual claim:
{original_claim}

Identify exactly ONE factual span in the claim that can be replaced
with a different but plausible false value.

The replacement must:
- be a noun, name, entity, location, relationship, occupation,
  attribute, or other factual element
- preserve the surrounding grammar
- not change any numbers
- not add or remove information
- not rewrite the sentence
- not change punctuation
- not change any other words

Return ONLY valid JSON with exactly these fields:

{{
  "target": "exact text copied from the original claim",
  "replacement": "plausible but false replacement text",
  "description": "briefly explain what factual element was changed"
}}

The target MUST appear exactly in the original claim.
The replacement must represent a false but plausible fact.
"""

    response = LLM.invoke(prompt)

    raw = response.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Factual corruption generator did not return valid JSON.\n"
            f"LLM output:\n{raw}"
        ) from exc

    target = str(data.get("target", "")).strip()
    replacement = str(data.get("replacement", "")).strip()
    description = str(data.get("description", "")).strip()

    if not target:
        raise ValueError(
            "Factual corruption generator returned an empty target."
        )

    if not replacement:
        raise ValueError(
            "Factual corruption generator returned an empty replacement."
        )

    if target not in original_claim:
        raise ValueError(
            "Factual corruption target does not occur exactly "
            "in the original claim.\n"
            f"Target: {target}\n"
            f"Original: {original_claim}"
        )

    if target == replacement:
        raise ValueError(
            "Factual corruption replacement is identical "
            "to the original target."
        )

    corrupted_claim = original_claim.replace(
        target,
        replacement,
        1,
    )

    return ErrorTemplate(
        error_type="factual",
        original_claim_pattern=original_claim,
        injected_claim=corrupted_claim,
        description=description,
    )


def _generate_numerical_corruption(
    original_claim: str,
) -> ErrorTemplate:
    numbers = _extract_numbers(original_claim)

    if not numbers:
        raise ValueError(
            "No suitable claim containing an existing numerical "
            "value was found in the Research Agent output."
        )

    prompt = f"""
You are creating a controlled numerical error for an experiment.

Original factual claim:
{original_claim}

The claim contains these numerical values:
{numbers}

Choose exactly ONE existing numerical value and change it to a
different but plausible false value.

Rules:
- change exactly one number
- do not change any other words
- do not add words
- do not remove words
- do not change punctuation
- the replacement must remain plausible
- return ONLY valid JSON

Format:

{{
  "original_number": "exact original number",
  "replacement_number": "new false number",
  "description": "briefly explain the numerical change"
}}
"""

    response = LLM.invoke(prompt)

    raw = response.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Numerical corruption generator did not return valid JSON.\n"
            f"LLM output:\n{raw}"
        ) from exc

    original_number = str(
        data.get("original_number", "")
    ).strip()

    replacement_number = str(
        data.get("replacement_number", "")
    ).strip()

    description = str(
        data.get("description", "")
    ).strip()

    if original_number not in numbers:
        raise ValueError(
            "Numerical generator selected a number that does not "
            "exist in the original claim."
        )

    if not replacement_number:
        raise ValueError(
            "Numerical generator returned an empty replacement number."
        )

    if original_number == replacement_number:
        raise ValueError(
            "Numerical generator did not actually change the number."
        )

    corrupted_claim = original_claim.replace(
        original_number,
        replacement_number,
        1,
    )

    return ErrorTemplate(
        error_type="numerical",
        original_claim_pattern=original_claim,
        injected_claim=corrupted_claim,
        description=description,
    )


def _generate_false_premise_corruption(
    original_claim: str,
) -> ErrorTemplate:
    prompt = f"""
You are creating a controlled false-premise error for an experiment.

Original factual claim:
{original_claim}

Create ONE additional false premise that makes the claim misleading
or causally incorrect.

The original factual content must remain unchanged.

Return ONLY valid JSON:

{{
  "false_premise": "one concise false premise",
  "description": "briefly explain why the premise is false"
}}

Do not rewrite the original claim.
Do not change existing facts.
Do not change numbers.
"""

    response = LLM.invoke(prompt)

    raw = response.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "False-premise generator did not return valid JSON.\n"
            f"LLM output:\n{raw}"
        ) from exc

    false_premise = str(
        data.get("false_premise", "")
    ).strip()

    description = str(
        data.get("description", "")
    ).strip()

    if not false_premise:
        raise ValueError(
            "False-premise generator returned an empty premise."
        )

    corrupted_claim = (
        f"{original_claim} This is because {false_premise}."
    )

    return ErrorTemplate(
        error_type="false_premise",
        original_claim_pattern=original_claim,
        injected_claim=corrupted_claim,
        description=description,
    )


def generate_corrupted_claim(
    research_output: str,
    error_type: str,
) -> ErrorTemplate:
    if not research_output:
        raise ValueError(
            "research_output cannot be empty."
        )

    if error_type not in {
        "factual",
        "numerical",
        "false_premise",
    }:
        raise ValueError(
            "Unsupported error type. "
            "Expected factual, numerical, or false_premise."
        )

    candidates = _extract_candidate_claims(
        research_output
    )

    if error_type == "factual":
        original_claim = candidates[0]

        template = _generate_factual_corruption(
            original_claim
        )

    elif error_type == "false_premise":
        original_claim = candidates[0]

        template = _generate_false_premise_corruption(
            original_claim
        )

    else:
        numerical_candidates = [
            candidate
            for candidate in candidates
            if _extract_numbers(candidate)
        ]

        if not numerical_candidates:
            raise ValueError(
                "No suitable claim containing an existing numerical "
                "value was found in the Research Agent output."
            )

        original_claim = numerical_candidates[0]

        template = _generate_numerical_corruption(
            original_claim
        )

    _validate_corruption(
        original_claim=template.original_claim_pattern,
        corrupted_claim=template.injected_claim,
        error_type=error_type,
    )

    return template