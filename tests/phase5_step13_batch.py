import json
import os
import random
import re
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from agents.research_agent import research_node
from injection.error_bank import (
    _extract_candidate_claims,
    _extract_numbers,
)
from pipelines.p1_linear import run_pipeline


RANDOM_SEED = 42
NUMBER_OF_QUESTIONS = 3

ERROR_TYPES = [
    "factual",
    "numerical",
    "false_premise",
]

EXPECTED_BASELINE_CORRECT = {
    "HQ00": True,
    "HQ01": True,
    "HQ02": True,
    "HQ03": True,
    "HQ04": True,
    "HQ05": True,
    "HQ06": True,
    "HQ07": False,
    "HQ08": False,
    "HQ09": True,
}


def load_tasks():
    with open("tasks/hotpotqa_sample.json", "r") as file:
        tasks = json.load(file)

    return tasks


def get_research_output(task):
    result = research_node(
        {
            "question": task["question"],
            "context": task["context"],
            "trace": [],
        }
    )

    return result["research_output"]["output"]


def extract_final_answer(final_output):
    """
    Extract only the answer text from the Action Agent's formatted output.

    Expected format:

        FINAL ANSWER: <answer>

        CONFIDENCE: <value>

    or:

        FINAL ANSWER: <answer>

        CONFIDENCE (0-1): <value>
    """

    if not isinstance(final_output, str):
        return ""

    answer_match = re.search(
        r"FINAL\s+ANSWER\s*:\s*(.*)",
        final_output,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not answer_match:
        return final_output.strip()

    answer_text = answer_match.group(1)

    confidence_match = re.search(
        r"\n\s*CONFIDENCE(?:\s*\(0-1\))?\s*:",
        answer_text,
        flags=re.IGNORECASE,
    )

    if confidence_match:
        answer_text = answer_text[
            :confidence_match.start()
        ]

    return answer_text.strip()


def normalize_answer(answer):
    """
    Deterministic normalization for clean-baseline comparison.

    The rule is:
    1. Lowercase.
    2. Collapse whitespace.
    3. Remove surrounding/trailing punctuation.
    4. Remove a leading 'the' article.
    5. Compare exact normalized strings OR accept clear
       containment in either direction, so harmless elaboration
       is accepted.
    """

    if not isinstance(answer, str):
        return ""

    normalized = answer.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)

    normalized = normalized.strip(
        " \t\r\n.,;:!?\"'“”‘’()[]{}"
    )

    normalized = re.sub(
        r"^the\s+",
        "",
        normalized,
    )

    normalized = normalized.strip(
        " \t\r\n.,;:!?\"'“”‘’()[]{}"
    )

    return normalized


def answers_match(extracted_answer, gold_answer):
    normalized_extracted = normalize_answer(
        extracted_answer
    )
    normalized_gold = normalize_answer(
        gold_answer
    )

    if not normalized_extracted or not normalized_gold:
        return False

    if normalized_extracted == normalized_gold:
        return True

    return normalized_gold in normalized_extracted


def get_clean_baseline(task):
    result = run_pipeline(
        task,
        inject=False,
        error_type=None,
        wording=None,
    )

    final_output = result["final_answer"].get(
        "output",
        "",
    )

    extracted_answer = extract_final_answer(
        final_output
    )

    gold_answer = task["gold_answer"]

    baseline_correct = answers_match(
        extracted_answer,
        gold_answer,
    )

    return {
        "result": result,
        "final_output": final_output,
        "extracted_answer": extracted_answer,
        "baseline_correct": baseline_correct,
    }


def validate_baseline_classification(tasks):
    print("\n" + "#" * 90)
    print("PHASE 5 STEP 13 — BASELINE NORMALIZER VALIDATION")
    print("#" * 90)

    actual_results = {}

    for task in tasks:
        baseline = get_clean_baseline(task)

        actual_results[task["task_id"]] = (
            baseline["baseline_correct"]
        )

        print(
            f"{task['task_id']}: "
            f"extracted={baseline['extracted_answer']!r} | "
            f"gold={task['gold_answer']!r} | "
            f"correct={baseline['baseline_correct']}"
        )

    if actual_results != EXPECTED_BASELINE_CORRECT:
        print("\nEXPECTED CLASSIFICATION:")
        for task_id, expected in EXPECTED_BASELINE_CORRECT.items():
            print(
                f"  {task_id}: "
                f"{'correct' if expected else 'incorrect'}"
            )

        print("\nACTUAL CLASSIFICATION:")
        for task_id in EXPECTED_BASELINE_CORRECT:
            actual = actual_results.get(task_id)
            print(
                f"  {task_id}: "
                f"{'correct' if actual else 'incorrect'}"
            )

        raise ValueError(
            "Baseline normalizer validation failed. "
            "The actual 10-question classification does "
            "not exactly match the required classification."
        )

    print("\nBASELINE NORMALIZER VALIDATION PASSED")
    print("Correct:   HQ00 HQ01 HQ02 HQ03 HQ04 HQ05 HQ06 HQ09")
    print("Incorrect: HQ07 HQ08")


def get_eligibility(task):
    research_output = get_research_output(task)

    candidates = _extract_candidate_claims(
        research_output
    )

    if not candidates:
        finding = ""
    else:
        finding = candidates[0]

    factual_eligible = bool(finding)

    false_premise_eligible = bool(finding)

    numerical_candidates = []

    for candidate in candidates:
        numbers = _extract_numbers(candidate)

        if numbers:
            numerical_candidates.append(
                {
                    "claim": candidate,
                    "numbers": numbers,
                }
            )

    numerical_eligible = bool(
        numerical_candidates
    )

    baseline = get_clean_baseline(task)

    return {
        "task": task,
        "research_output": research_output,
        "factual": factual_eligible,
        "numerical": numerical_eligible,
        "false_premise": false_premise_eligible,
        "numerical_candidates": numerical_candidates,
        "baseline_correct": baseline[
            "baseline_correct"
        ],
        "baseline_output": baseline[
            "final_output"
        ],
        "baseline_extracted_answer": baseline[
            "extracted_answer"
        ],
    }


def select_questions(tasks):
    eligibility_results = []

    print("\n" + "#" * 90)
    print("PHASE 5 STEP 13 — ELIGIBILITY + BASELINE SCREEN")
    print("#" * 90)

    for task in tasks:
        result = get_eligibility(task)
        eligibility_results.append(result)

        print("\n" + "-" * 90)
        print(f"TASK: {task['task_id']}")
        print(
            f"Factual:          "
            f"{result['factual']}"
        )
        print(
            f"Numerical:        "
            f"{result['numerical']}"
        )
        print(
            f"False premise:    "
            f"{result['false_premise']}"
        )
        print(
            f"Clean baseline:   "
            f"{result['baseline_correct']}"
        )
        print(
            f"Extracted answer: "
            f"{result['baseline_extracted_answer']!r}"
        )
        print(
            f"Gold answer:      "
            f"{task['gold_answer']!r}"
        )

        if result["numerical_candidates"]:
            print("Numerical candidates:")

            for candidate in result[
                "numerical_candidates"
            ]:
                print(
                    f"  {candidate['numbers']} "
                    f"-> {candidate['claim']}"
                )

    common_eligible = [
        result
        for result in eligibility_results
        if result["factual"]
        and result["numerical"]
        and result["false_premise"]
        and result["baseline_correct"]
    ]

    print("\n" + "#" * 90)
    print("COMMON ELIGIBLE QUESTIONS")
    print("#" * 90)

    for result in common_eligible:
        print(
            f"{result['task']['task_id']} | "
            f"{result['task']['question']}"
        )

    if len(common_eligible) != NUMBER_OF_QUESTIONS:
        raise ValueError(
            "Expected exactly 3 questions to be eligible "
            "for all three starter error types with a "
            "correct clean baseline, but found "
            f"{len(common_eligible)}."
        )

    selected_results = sorted(
        common_eligible,
        key=lambda result: result["task"]["task_id"],
    )

    print("\n" + "#" * 90)
    print(
        f"SELECTED {NUMBER_OF_QUESTIONS} QUESTIONS"
    )
    print("#" * 90)

    for result in selected_results:
        print(
            f"{result['task']['task_id']} | "
            f"{result['task']['question']}"
        )

    return [
        result["task"]
        for result in selected_results
    ]


def run_control(task):
    return run_pipeline(
        task,
        inject=False,
        error_type=None,
        wording=None,
    )


def run_treatment(task, error_type):
    return run_pipeline(
        task,
        inject=True,
        error_type=error_type,
        wording="neutral",
    )


def print_result(
    task,
    condition,
    result,
):
    final_answer = result["final_answer"]

    print("\n" + "=" * 90)
    print(f"TASK: {task['task_id']}")
    print(f"QUESTION: {task['question']}")
    print(f"GOLD ANSWER: {task['gold_answer']}")
    print(f"CONDITION: {condition}")

    print("\n--- RESEARCH OUTPUT ---")
    print(
        result["research_output"].get(
            "output",
            "",
        )
    )

    if condition != "clean":
        print("\n--- INJECTOR ---")

        injector_entries = [
            entry
            for entry in result["trace"]
            if entry.get("agent") == "error_injector"
        ]

        if injector_entries:
            print(
                injector_entries[0]["output"]
            )

    print("\n--- SUMMARY OUTPUT ---")
    print(
        result["summary"].get(
            "output",
            "",
        )
    )

    print("\n--- FINAL ANSWER ---")
    print(
        final_answer.get(
            "output",
            "",
        )
    )

    print("=" * 90)


def main():
    tasks = load_tasks()

    print(
        f"Loaded {len(tasks)} total questions."
    )

    # Step 13 established selection.
    selected_ids = {"HQ02", "HQ05", "HQ06"}

    selected_tasks = sorted(
        [
            task
            for task in tasks
            if task["task_id"] in selected_ids
        ],
        key=lambda task: task["task_id"],
    )

    if len(selected_tasks) != NUMBER_OF_QUESTIONS:
        raise ValueError(
            "Expected exactly HQ02, HQ05, HQ06 for Step 13."
        )

    print("\n" + "#" * 90)
    print("PHASE 5 STEP 13 — EXPERIMENT")
    print("#" * 90)

    print("\nSELECTED QUESTIONS")
    for task in selected_tasks:
        print(
            f"{task['task_id']} | "
            f"{task['question']}"
        )

    total_runs = 0

    for task in selected_tasks:
        control_result = run_control(task)

        print_result(
            task=task,
            condition="clean",
            result=control_result,
        )

        total_runs += 1

        for error_type in ERROR_TYPES:
            treatment_result = run_treatment(
                task,
                error_type,
            )

            print_result(
                task=task,
                condition=f"{error_type} / neutral",
                result=treatment_result,
            )

            total_runs += 1

    expected_runs = NUMBER_OF_QUESTIONS * (1 + len(ERROR_TYPES))

    if total_runs != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs, "
            f"but executed {total_runs}."
        )

    print("\n" + "#" * 90)
    print(
        f"PHASE 5 STEP 13 COMPLETE: "
        f"{total_runs} runs executed."
    )
    print("#" * 90)


if __name__ == "__main__":
    main()
