# Silent Compounding: Diagnosing Error Cascades in Multi-Agent LLM Systems

**Research Project — Data Science / LLM Systems**

Silent Compounding is a research project investigating how errors introduced into one stage of a multi-agent Large Language Model (LLM) system can propagate through subsequent agents and eventually influence the final answer.

The central question is simple:

> **When one agent makes a mistake, what happens to that mistake as it moves through the rest of the system?**

Rather than looking only at whether the final answer is correct or incorrect, this project studies the **propagation of errors across intermediate stages** of a multi-agent pipeline.

---

## Overview

Multi-agent LLM systems often divide a task across several agents. For example:

```text
Research Agent
      ↓
Summarizer Agent
      ↓
Action Agent
      ↓
Final Answer
```

This decomposition can make complex tasks easier to handle, but it also introduces another problem: an incorrect claim produced early in the pipeline may be accepted by later agents and become part of their reasoning.

The project therefore treats an LLM pipeline as a sequence in which an error can potentially:

* disappear,
* remain unchanged,
* be rephrased,
* become more strongly stated,
* or influence the final answer.

The goal is to measure and diagnose these patterns systematically.

---

## Research Objectives

The project currently focuses on four main objectives:

1. **Establish a baseline** for a multi-agent LLM pipeline using clean inputs.
2. **Inject controlled errors** at an early stage of the pipeline.
3. **Trace whether those errors survive** through downstream agents.
4. **Compare automated error detection with manually labelled ground truth.**

The longer-term goal is to understand which types of errors are most likely to compound and under what conditions downstream agents fail to correct them.

---

## Error Taxonomy

The project uses six broad categories of injected errors:

| Code | Error Type           | Description                                               |
| ---- | -------------------- | --------------------------------------------------------- |
| E1   | Factual substitution | Replacing a correct fact with an incorrect fact           |
| E2   | Numerical corruption | Changing a numerical value                                |
| E3   | Entity substitution  | Replacing one entity with another                         |
| E4   | Temporal error       | Introducing an incorrect date or time                     |
| E5   | Relationship error   | Changing a relationship between entities or facts         |
| E6   | False premise        | Introducing a premise that is not supported by the source |

The initial experimental work concentrates on **factual substitution, numerical corruption, and false-premise errors**.

---

## Experimental Design

The project uses controlled error injection so that the original claim and the corrupted claim are both known.

This allows the experiment to distinguish between:

```text
Clean output
     ↓
Controlled error injection
     ↓
Downstream agents
     ↓
Error tracing
     ↓
Final result
```

Different forms of wording are also being investigated, including neutral, confident, hedged, evidence-backed, and instruction-backed formulations.

This makes it possible to study not only **what error was introduced**, but also whether the way an error is presented affects its propagation.

---

## Current Pipeline

The initial pipeline is implemented using **LangGraph**:

```text
HotpotQA Task
     │
     ▼
Research Agent
     │
     ▼
Summarizer Agent
     │
     ▼
Action Agent
     │
     ▼
Final Answer
```

The Research Agent works from the supplied HotpotQA context. The downstream agents process the research output and produce a summary and final answer.

The baseline pipeline has been evaluated on a 10-question sample before introducing controlled error injection.

---

## Error Injection

Error injection is performed at the research stage of the pipeline.

The injector:

1. Identifies a suitable claim in the research output.
2. Generates a controlled corrupted version of that claim.
3. Applies the selected wording condition.
4. Replaces the original claim while preserving the surrounding output.
5. Keeps the original and injected claims available as experimental ground truth.

This allows downstream outputs to be compared against a known injected error rather than relying on subjective interpretation of what the original error was.

---

## Error Tracing

The next stage of the project focuses on automatically determining whether an injected error is still present downstream.

Two approaches are being developed:

### Rule-Based Tracing

A lightweight detector checks whether the injected claim, entity, or numerical value appears in downstream text.

### LLM-as-Judge

An LLM is asked whether the downstream text still asserts the injected claim, including cases where the claim has been reworded.

The two approaches will be compared against manually labelled ground truth.

---

## Dataset

The project currently uses a sample of **HotpotQA**, a multi-hop question answering dataset.

The dataset provides:

* questions,
* answers,
* supporting context,
* and supporting facts.

The controlled experimental setup uses these known source facts to construct and evaluate error propagation scenarios.

---

## Project Structure

```text
silent-compounding/
│
├── agents/
│   ├── research_agent.py
│   ├── summarizer_agent.py
│   ├── action_agent.py
│   ├── critic_agent.py
│   └── verifier_agent.py
│
├── analysis/
│   ├── metrics.py
│   ├── plots.py
│   └── notebook.ipynb
│
├── injection/
│   ├── error_bank.py
│   ├── injector.py
│   └── wording.py
│
├── pipelines/
│   ├── p1_linear.py
│   ├── p2_verifier.py
│   ├── p3_parallel.py
│   ├── p4_debate.py
│   ├── p5_supervisor.py
│   └── p6_governance.py
│
├── tasks/
│   ├── hotpotqa_sample.json
│   └── custom_tasks.json
│
├── tracing/
│   ├── logger.py
│   └── judge.py
│
├── data/
│   ├── experiment_log.csv
│   └── raw_runs/
│
├── tests/
│
├── run_experiment.py
├── config.yaml
├── requirements.txt
└── README.md
```

Some components in the repository are part of the planned experimental architecture and are being implemented progressively.

---

## Technology Stack

* **Python 3.12**
* **LangGraph**
* **LangChain**
* **Ollama**
* **Llama 3.1**
* **Mistral**
* **Pandas**
* **SciPy**
* **Matplotlib**
* **Seaborn**
* **Jupyter**
* **Git / GitHub**

The current experiments use locally hosted LLMs through Ollama.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/shlok2107-ui/silent-compounding.git
cd silent-compounding
```

### 2. Create / activate the environment

The project is developed using a dedicated Conda environment with Python 3.12.

```bash
conda activate silent-compounding
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Make sure Ollama is available

The experiments currently use local models through Ollama.

For example:

```bash
ollama list
```

The current development environment uses Llama 3.1 and Mistral.

### 5. Run the pipeline

The project is being developed incrementally by research phase. Individual pipeline and experiment scripts can be run from the project root as they are completed.

---

## Experiment Logging

The project records experimental information including:

* run ID,
* pipeline,
* model,
* task ID,
* error type,
* injection stage,
* wording condition,
* detected error,
* correction stage,
* final error status,
* severity,
* self-reported confidence,
* actual correctness,
* and raw trace location.

Raw traces are stored separately so that the intermediate outputs can be inspected rather than relying only on final metrics.

---

## Research Direction

The planned progression is:

```text
Baseline Pipeline
       ↓
Controlled Error Injection
       ↓
Error Tracing
       ↓
Manual Ground Truth
       ↓
Tracer Validation
       ↓
Multiple Multi-Agent Architectures
       ↓
Error Propagation Metrics
       ↓
Cross-Architecture Comparison
```

Future experiments will examine whether different multi-agent architectures behave differently when exposed to the same injected errors.

The eventual objective is to move from simply asking:

> "Was the final answer correct?"

towards questions such as:

> "Where did the error originate?"

> "Did downstream agents detect it?"

> "At which stage did it disappear or become reinforced?"

> "Which types of errors are most likely to survive through the system?"

---

## Project Status

**Status: Active Research / In Development**

The baseline pipeline, experiment logging, controlled error injection, and initial treatment runs have been implemented. The project is currently moving into the error-tracing and validation stage.

Results and experimental conclusions will be added as the study progresses.

---

## Author

**Shlok Agrawal**

Research project in Data Science / Large Language Model Systems.

---

## Note

This repository contains an ongoing research implementation. Experimental results, pipeline architectures, and analysis may change as the study progresses and additional validation is performed.
