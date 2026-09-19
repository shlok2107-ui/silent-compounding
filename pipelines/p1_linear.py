import json
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langgraph.graph import END, StateGraph

from agents.research_agent import research_node
from agents.summarizer_agent import summarize_node
from agents.action_agent import action_node
from tracing.logger import log_run, save_raw_trace


class PipelineState(TypedDict):
    question: str
    context: dict
    research_output: dict
    summary: dict
    final_answer: dict
    trace: list


def summarize_node_adapter(state):
    result = summarize_node(
        {
            "research_output": state["research_output"],
            "trace": state["trace"],
        }
    )

    return {
        "summary": result["summary_output"],
        "trace": result["trace"],
    }


def action_node_adapter(state):
    result = action_node(
        {
            "question": state["question"],
            "summary_output": state["summary"],
            "trace": state["trace"],
        }
    )

    return {
        "final_answer": result["action_output"],
        "trace": result["trace"],
    }


graph = StateGraph(PipelineState)

graph.add_node("research", research_node)
graph.add_node("summarize", summarize_node_adapter)
graph.add_node("action", action_node_adapter)

graph.set_entry_point("research")

graph.add_edge("research", "summarize")
graph.add_edge("summarize", "action")
graph.add_edge("action", END)

pipeline = graph.compile()


if __name__ == "__main__":
    with open("tasks/hotpotqa_sample.json", "r") as file:
        tasks = json.load(file)

    print(f"Loaded {len(tasks)} questions.")

    for task in tasks:
        initial_state = {
            "question": task["question"],
            "context": task["context"],
            "research_output": {},
            "summary": {},
            "final_answer": {},
            "trace": [],
        }

        final_state = pipeline.invoke(initial_state)

        run_id = task["task_id"]

        raw_trace_path = save_raw_trace(
            run_id,
            final_state["trace"],
        )

        final_answer = final_state["final_answer"]

        log_run(
            {
                "run_id": run_id,
                "pipeline": "P1",
                "model": "llama3.1",
                "task_id": task["task_id"],
                "error_type": None,
                "injection_stage": None,
                "wording": None,
                "error_detected": None,
                "correction_stage": None,
                "final_error": None,
                "severity": None,
                "self_reported_confidence": final_answer.get(
                    "confidence",
                    None,
                ),
                "actual_correct": None,
                "raw_trace_path": raw_trace_path,
            }
        )

        print("\n" + "=" * 80)
        print(f"TASK: {task['task_id']}")
        print(f"QUESTION: {task['question']}")
        print(f"GOLD ANSWER: {task['gold_answer']}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"RAW TRACE: {raw_trace_path}")
        print("=" * 80)