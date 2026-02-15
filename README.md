# Multi-Agent Orchestration with Observability

## Overview
This project demonstrates a production-style multi-agent system built using locally hosted LLMs via Ollama, with integrated observability using Phoenix (Arize) and a Streamlit-based monitoring dashboard.

The system simulates how multiple AI agents collaborate to solve complex tasks while providing full visibility into their reasoning, execution flow, and performance.

This project focuses on building reliable, inspectable, and debuggable agentic systems rather than just generating outputs.

---

## Key Features

- Multi-agent orchestration with clear separation of responsibilities  
- Local LLM inference using Ollama (no external API cost)  
- End-to-end observability using Phoenix (Arize)  
- Interactive dashboard built with Streamlit  
- Real-time tracing of agent decisions and tool usage  
- Evaluation and metrics generation for agent performance  
- Support for both simulated and real search workflows  

---

## Architecture

The system consists of multiple components working together:

### 1. Agents
Each agent is responsible for a specific task such as:
- Query understanding  
- Research generation  
- Result synthesis  
- Evaluation  

Agents collaborate in a pipeline to produce the final output.

### 2. Orchestrator
The orchestrator coordinates the agents, manages the workflow, and ensures that outputs from one agent become inputs to the next.

### 3. LLM Backend (Ollama)
All language model calls are executed l
