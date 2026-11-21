"""Pure, dependency-free chunking helpers (no models, no network, no PDFs).

Kept separate so they can be unit-tested offline and in CI.
"""


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        # Without this guard the window never advances (infinite loop).
        raise ValueError("overlap must be smaller than chunk_size")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into word windows of `chunk_size` words overlapping by `overlap`."""
    _validate(chunk_size, overlap)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break  # last window reached the end; avoid a redundant tail chunk
        start += chunk_size - overlap
    return chunks


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Fixed-size word chunks returned as dicts with id/strategy/word_count."""
    return [
        {
            "id": f"fixed_{i}",
            "text": c,
            "strategy": "fixed_size",
            "word_count": len(c.split()),
        }
        for i, c in enumerate(chunk_text(text, chunk_size, overlap))
    ]
