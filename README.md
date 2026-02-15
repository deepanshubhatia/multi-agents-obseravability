# Multi-Agent Orchestration with Observability

## Overview

This project demonstrates a production-style multi-agent system built using locally hosted LLMs via Ollama, with integrated observability using Phoenix (Arize) and a Streamlit-based monitoring dashboard.

The system simulates how multiple AI agents collaborate to solve complex tasks while providing full visibility into their reasoning, execution flow, and performance.

This project focuses on building reliable, inspectable, and debuggable agentic systems rather than just generating outputs.

---

## Key Features

* Multi-agent orchestration with clear separation of responsibilities
* Local LLM inference using Ollama (no external API cost)
* End-to-end observability using Phoenix (Arize)
* Interactive dashboard built with Streamlit
* Real-time tracing of agent decisions and tool usage
* Evaluation and metrics generation for agent performance
* Support for both simulated and real search workflows

---

## Architecture

The system consists of multiple components working together:

### 1. Agents

Each agent is responsible for a specific task such as:

* Query understanding
* Research generation
* Result synthesis
* Evaluation

Agents collaborate in a pipeline to produce the final output.

### 2. Orchestrator

The orchestrator coordinates the agents, manages the workflow, and ensures that outputs from one agent become inputs to the next.

### 3. LLM Backend (Ollama)

All language model calls are executed locally using Ollama, allowing:

* Cost-free experimentation
* Full control over models
* Offline capability

### 4. Observability (Phoenix - Arize)

Phoenix is used to capture:

* Agent traces
* LLM inputs and outputs
* Latency and performance metrics
* Error analysis

This enables debugging and improving agent behavior.

### 5. Dashboard (Streamlit)

A local dashboard is provided to visualize:

* Execution traces
* Agent interactions
* Metrics and evaluation results

---

## Project Structure

```
multi-agents-observability/
│
├── src/                       # Core multi-agent logic
├── evaluation_output/         # Evaluation results
├── observability_data/        # Phoenix trace data
├── research_outputs/          # Generated research outputs
│
├── demo.py                    # Basic demo pipeline
├── demo_real_search.py        # Real-world search demo
├── real_research_demo.py      # Multi-agent research workflow
├── generate_real_research.py  # Generates research outputs
├── generate_metrics.py        # Evaluation and metrics
├── observability_dashboard.py # Streamlit dashboard
├── server.py                  # Backend server
│
├── test_multisearch.py        # Tests for multi-search
├── test_real_search.py        # Tests for real search
│
└── requirements.txt           # Dependencies
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/deepanshubhatia/multi-agents-obseravability.git
cd multi-agents-obseravability
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and Run Ollama

Follow instructions from https://ollama.com

Pull a model:

```bash
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

---

## Running the Project

### Run a Basic Demo

```bash
python demo.py
```

### Run Real Search Workflow

```bash
python demo_real_search.py
```

### Generate Research Outputs

```bash
python generate_real_research.py
```

### Generate Metrics

```bash
python generate_metrics.py
```

---

## Observability Dashboard

To start the Streamlit dashboard:

```bash
streamlit run observability_dashboard.py
```

This will open a local UI where you can:

* Inspect agent traces
* Analyze model responses
* Debug failures
* View performance metrics

---

## Phoenix (Arize) Integration

Phoenix is used to capture detailed traces of the agent workflow.

Key benefits:

* Understand how agents reason
* Identify bottlenecks
* Debug incorrect outputs
* Evaluate system quality

Ensure Phoenix is running locally or configured properly before executing the pipelines.

---

## Example Use Case

1. User provides a query
2. Query agent interprets the request
3. Research agent gathers relevant information
4. Synthesis agent composes the final answer
5. Evaluation agent scores the output
6. All steps are logged and visualized via Phoenix and Streamlit

---

## Why This Project Matters

Most AI demos focus only on output quality. This project emphasizes:

* Observability of AI systems
* Debugging multi-agent pipelines
* Production readiness of LLM applications
* Cost-efficient local deployment

This is critical for real-world enterprise AI systems.

---

## Future Improvements

* Add memory across agent interactions
* Integrate external tools and APIs
* Support for multiple LLM providers
* Advanced evaluation techniques
* Kubernetes-based deployment

---

## License

MIT License

---

## Author

Deepanshu Bhatia
