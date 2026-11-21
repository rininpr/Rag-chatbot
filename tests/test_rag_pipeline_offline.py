"""Contract tests for rag_query with retrieval and the LLM mocked.

Deterministic, no network, no API keys: these run in CI on every push.
The live faithfulness evaluation is in test_faithfulness.py (opt-in).
"""


def test_result_contract(wired):
    out = wired.rag_query("What chip does Mac Studio use?")
    assert set(out) == {"answer", "chunks", "distances", "embed_ms", "retrieve_ms",
                        "gen_ms", "total_ms", "tokens"}
    assert out["tokens"] == 120
    assert len(out["chunks"]) == len(out["distances"]) == 3


def test_answer_is_stripped(wired):
    assert wired.rag_query("q")["answer"] == "The M4 Max."


def test_prompt_contains_all_chunks_and_question(wired):
    wired.rag_query("What chip does Mac Studio use?")
    prompt = wired.groq_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "What chip does Mac Studio use?" in prompt
    for chunk in wired.collection.query.return_value["documents"][0]:
        assert chunk in prompt


def test_prompt_enforces_grounding_and_fallback_message(wired):
    wired.rag_query("anything")
    prompt = wired.groq_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Answer ONLY using the context" in prompt
    assert "I don't have that information in the Mac Studio documentation." in prompt


def test_generation_is_low_temperature_and_bounded(wired):
    wired.rag_query("q")
    kw = wired.groq_client.chat.completions.create.call_args.kwargs
    assert kw["temperature"] <= 0.2
    assert kw["max_tokens"] == 500


def test_retrieval_requests_at_most_three_chunks(wired):
    wired.rag_query("q")
    assert wired.collection.query.call_args.kwargs["n_results"] == 3


def test_retrieval_clamps_to_collection_size(wired):
    wired.collection.count.return_value = 2
    wired.rag_query("q")
    assert wired.collection.query.call_args.kwargs["n_results"] == 2


def test_llm_failure_propagates_not_swallowed(wired):
    import pytest
    wired.groq_client.chat.completions.create.side_effect = RuntimeError("rate limited")
    with pytest.raises(RuntimeError):
        wired.rag_query("q")
