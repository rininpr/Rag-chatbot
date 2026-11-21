# RAG Chatbot

Document Q&A over PDFs.

- `ingest.py` - load documents, chunk, embed into ChromaDB
- `chunking_strategies.py` - chunking approaches compared
- `chatbot.py` - retrieval + Groq LLM answers, Langfuse tracing
- `tests/test_faithfulness.py` - DeepEval faithfulness test

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q          # 20 offline tests, <1s, no keys needed
```
Live LLM faithfulness eval is opt-in; see [TESTING.md](TESTING.md) for the strategy.

## Setup
Create a local `.env` (not committed) with `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`. `pip install -r requirements.txt`. Put the PDF in `documents/` (or set `PDF_PATH`), run `python ingest.py`, then `python chatbot.py`.
