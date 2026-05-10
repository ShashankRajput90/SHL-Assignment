---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# SHL Assessment Recommender

A conversational AI agent that helps hiring managers and recruiters find the right SHL assessments through dialogue.

## What it does

Takes a vague hiring intent ("I am hiring a Java developer") and guides the user to a grounded shortlist of SHL Individual Test Solutions through multi-turn conversation.

**Four core behaviors:**

- **Clarify** — asks focused questions when the query is too vague
- **Recommend** — returns 1-10 assessments with names and catalog URLs
- **Refine** — updates the shortlist when the user changes constraints
- **Compare** — answers comparison questions using catalog data only

## API Endpoints

### GET /health

Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

Takes full conversation history, returns agent reply and optional recommendations.

**Request:**

```json
{
  "messages": [
    { "role": "user", "content": "I am hiring a mid-level Java developer" }
  ]
}
```

**Response:**

```json
{
  "reply": "Here are assessments that fit a mid-level Java developer.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Tech Stack

- **FastAPI** — API framework
- **Groq (llama-3.3-70b-versatile)** — LLM for conversation and reasoning
- **FAISS + sentence-transformers** — vector search over SHL catalog
- **HuggingFace Spaces (Docker)** — deployment

## Project Structure

```
├── app/
│   ├── main.py         # FastAPI endpoints
│   ├── agent.py        # LLM call and response parsing
│   └── retrieval.py    # FAISS index and search
├── data/
│   └── shl_product_catalog.json
├── Dockerfile
└── requirements.txt
```
