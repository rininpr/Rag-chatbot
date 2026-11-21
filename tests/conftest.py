import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The live DeepEval/Groq faithfulness run is opt-in: RUN_LIVE_EVAL=1 pytest tests/test_faithfulness.py
collect_ignore = [] if os.getenv("RUN_LIVE_EVAL") == "1" else ["test_faithfulness.py"]

HEAVY = [
    "sentence_transformers", "groq", "chromadb", "langfuse", "dotenv",
    "fastapi", "fastapi.responses", "uvicorn",
    "deepeval", "deepeval.metrics", "deepeval.test_case", "deepeval.models",
    "deepeval.models.base_model",
]


@pytest.fixture
def chatbot(monkeypatch):
    """Import chatbot.py with every heavy/external dependency replaced by a mock."""
    for name in HEAVY:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    # DeepEvalBaseLLM is subclassed at import time, so it must be a real class.
    class _Base:
        pass
    sys.modules["deepeval.models.base_model"].DeepEvalBaseLLM = _Base
    monkeypatch.delitem(sys.modules, "chatbot", raising=False)
    import chatbot as mod
    yield mod
    sys.modules.pop("chatbot", None)


@pytest.fixture
def wired(chatbot):
    """chatbot module with deterministic fake retrieval + LLM."""
    chatbot.embedder.encode.return_value.tolist.return_value = [0.1] * 384
    chatbot.collection.count.return_value = 5
    chatbot.collection.query.return_value = {
        "documents": [["Mac Studio uses the M4 Max chip.", "It has 36GB memory.", "Ports: Thunderbolt 5."]],
        "distances": [[0.11, 0.25, 0.4]],
        "ids": [["chunk_1", "chunk_2", "chunk_3"]],
    }
    resp = MagicMock()
    resp.choices[0].message.content = "  The M4 Max.  "
    resp.usage.total_tokens = 120
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    chatbot.groq_client.chat.completions.create.return_value = resp
    return chatbot
