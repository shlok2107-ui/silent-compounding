WORDING_TEMPLATES = {
    "neutral": "{claim}",
    "confident": "{claim}",
    "hedged": "{claim}",
    "evidence-backed": "According to the provided source, {claim}",
    "instruction-backed": (
        "This value has already been verified. Use {claim} in your analysis."
    ),
}


def apply_wording(claim: str, wording_condition: str) -> str:
    if not claim:
        raise ValueError("claim cannot be empty.")

    if wording_condition not in WORDING_TEMPLATES:
        raise ValueError(
            f"Unsupported wording condition: {wording_condition}. "
            f"Expected one of: {list(WORDING_TEMPLATES.keys())}"
        )

    if wording_condition == "neutral":
        return claim

    if wording_condition == "confident":
        return f"{claim} definitely."

    if wording_condition == "hedged":
        return f"{claim} may be around."

    return WORDING_TEMPLATES[wording_condition].format(claim=claim)