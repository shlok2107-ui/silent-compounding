from injection.error_bank import generate_corrupted_claim
from injection.wording import apply_wording


def inject_error(
    clean_output: str,
    error_type: str,
    wording: str,
):
    if not clean_output:
        raise ValueError("clean_output cannot be empty.")

    if not error_type:
        raise ValueError("error_type cannot be empty.")

    if not wording:
        raise ValueError("wording cannot be empty.")

    error_template = generate_corrupted_claim(
        research_output=clean_output,
        error_type=error_type,
    )

    original_claim = error_template.original_claim_pattern
    corrupted_claim = apply_wording(
        error_template.injected_claim,
        wording,
    )

    if original_claim not in clean_output:
        raise ValueError(
            "The generated original claim was not found in the clean output."
        )

    corrupted_output = clean_output.replace(
        original_claim,
        corrupted_claim,
        1,
    )

    return corrupted_output, error_template