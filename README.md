# RAG Chatbot

Document Q&A over PDFs.

- `ingest.py` - load documents, chunk, embed into ChromaDB
- `chunking_strategies.py` - chunking approaches compared
- `chatbot.py` - retrieval + Groq LLM answers, Langfuse tracing
- `tests/test_faithfulness.py` - DeepEval faithfulness test

## Setup
Create a local `.env` (not committed) with `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`. Put PDFs in `documents/`, run `python ingest.py`, then `python chatbot.py`.
