# SHL Assessment Recommendation Agent

A conversational AI agent that helps hiring managers find relevant SHL assessments.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python scraper/scrape_catalog.py
python embeddings/build_index.py
uvicorn app.main:app --reload
```

Then open: http://localhost:8000/docs

## API

- `GET /health` — health check
- `POST /chat` — conversational assessment recommender

## Deployment

Deployed on Railway. See `railway.json` for config.
