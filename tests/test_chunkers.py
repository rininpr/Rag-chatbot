import pytest

from chunkers import chunk_text, fixed_size_chunking

WORDS = [f"w{i}" for i in range(1000)]
TEXT = " ".join(WORDS)


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("a b c", chunk_size=10, overlap=2) == ["a b c"]


def test_chunk_size_never_exceeded():
    for c in chunk_text(TEXT, chunk_size=100, overlap=10):
        assert len(c.split()) <= 100


def test_overlap_is_exact():
    chunks = chunk_text(TEXT, chunk_size=100, overlap=10)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-10:] == nxt.split()[:10]


def test_no_word_is_lost():
    seen = set()
    for c in chunk_text(TEXT, chunk_size=100, overlap=10):
        seen.update(c.split())
    assert seen == set(WORDS)


def test_no_redundant_tail_chunk():
    # 500 words, size 500: the old implementation emitted a second 50-word duplicate.
    assert len(chunk_text(" ".join(WORDS[:500]), 500, 50)) == 1


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 11), (0, 0), (-1, 0), (10, -1)])
def test_invalid_params_raise_instead_of_hanging(size, overlap):
    with pytest.raises(ValueError):
        chunk_text(TEXT, size, overlap)


def test_fixed_size_chunking_schema_and_unique_ids():
    out = fixed_size_chunking(TEXT, 100, 10)
    assert len(out) == len({c["id"] for c in out})
    for c in out:
        assert set(c) == {"id", "text", "strategy", "word_count"}
        assert c["strategy"] == "fixed_size"
        assert c["word_count"] == len(c["text"].split())
