import csv
import json
import os


CSV_PATH = "data/experiment_log.csv"

CSV_COLUMNS = [
    "run_id",
    "pipeline",
    "model",
    "task_id",
    "error_type",
    "injection_stage",
    "wording",
    "error_detected",
    "correction_stage",
    "final_error",
    "severity",
    "self_reported_confidence",
    "actual_correct",
    "raw_trace_path",
]


def log_run(row: dict):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists or os.path.getsize(CSV_PATH) == 0:
            writer.writeheader()

        writer.writerow(row)


def save_raw_trace(run_id, trace):
    raw_runs_directory = "data/raw_runs"

    os.makedirs(raw_runs_directory, exist_ok=True)

    trace_path = os.path.join(
        raw_runs_directory,
        f"run_{run_id}.json",
    )

    with open(trace_path, "w", encoding="utf-8") as file:
        json.dump(
            trace,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return trace_path