# 07 — LLM Agents

An agent that recommends a medical clinic from a free-text request — "I'm looking for a clinic in Ursynów with a diabetologist" — by combining structured data, patient reviews and a local language model. Built as an explicit **LangGraph** state machine rather than a free-running ReAct loop.

## The graph

```
parse_request → structured_search → reviews_retrieval → quality_filter → ranking → answer
```

Each node has one job and writes into a typed `AgentState`:

| Node | Does |
|---|---|
| `parse_request` | LLM extracts problem, location and preferences from free text |
| `structured_search` | deterministic filter over the clinic table (specialty, district, rating) |
| `reviews_retrieval` | embedding search over patient reviews for the shortlisted clinics |
| `quality_filter` | drops candidates below rating/review-count thresholds, relaxing them if fewer than three survive |
| `ranking` | combines structured score and review evidence |
| `answer` | LLM writes the recommendation from the ranked list |

## Why a graph rather than an agent loop

The steps are known in advance, so letting a model decide the control flow buys nothing and costs determinism. Fixing the graph means the retrieval always runs before ranking, the filter always runs before the answer, and a failure is attributable to a node. The LLM is used where it is actually needed — parsing unstructured input and writing prose — and not for tasks a `pandas` filter does correctly every time.

The symptom-to-specialty mapping is a deliberate heuristic dictionary; the notebook's closing section lists it as a limitation, along with the absence of real geolocation, and suggests a classifier as the successor.

## Stack

LangGraph (`StateGraph`), LangChain (`StructuredTool`, `Document`, prompt templates), `Qwen2.5-3B-Instruct` served locally through a `transformers` pipeline.

## Run

`clinics.csv` and `clinic_reviews.csv` were provided by the course and are not redistributed here — place them beside the notebook. Then run on Colab with a GPU.
