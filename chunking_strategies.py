import os
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv("/mnt/HC_Volume_106169372/rag/.env")

PDF_PATH = "/mnt/HC_Volume_106169372/rag/documents/Introducing Mac Studio 2025 (en_GB).pdf"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text() -> str:
    reader = PdfReader(PDF_PATH)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

# ── STRATEGY 1: FIXED SIZE ──────────────────────────────────────────
def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": f"fixed_{index}",
            "text": chunk_text,
            "strategy": "fixed_size",
            "word_count": len(chunk_text.split())
        })
        start += chunk_size - overlap
        index += 1
    return chunks

# ── STRATEGY 2: RECURSIVE ───────────────────────────────────────────
def recursive_chunking(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list:
    separators = ["\n\n", "\n", ". ", " "]

    def split_text(text, separators):
        if not separators:
            return [text]
        sep = separators[0]
        splits = text.split(sep)
        chunks = []
        current = ""
        for split in splits:
            candidate = current + (sep if current else "") + split
            if len(candidate.split()) <= max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(split.split()) > max_chunk_size:
                    sub_chunks = split_text(split, separators[1:])
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                else:
                    current = split
        if current:
            chunks.append(current.strip())
        return chunks

    raw_chunks = split_text(text, separators)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_words = raw_chunks[i-1].split()[-overlap:]
            chunk = " ".join(prev_words) + " " + chunk
        chunks.append({
            "id": f"recursive_{i}",
            "text": chunk,
            "strategy": "recursive",
            "word_count": len(chunk.split())
        })
    return chunks

# ── STRATEGY 3: SEMANTIC ────────────────────────────────────────────
def semantic_chunking(text: str, threshold: float = 0.4) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return [{"id": "semantic_0", "text": text, "strategy": "semantic",
                 "word_count": len(text.split())}]

    print(f"  Embedding {len(sentences)} sentences...")
    embeddings = embedder.encode(sentences)

    similarities = []
    for i in range(len(embeddings) - 1):
        cos_sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        similarities.append(float(cos_sim))

    split_points = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.append(i + 1)
            print(f"  Topic shift at sentence {i+1} (similarity={sim:.3f})")
    split_points.append(len(sentences))

    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i+1]
        chunk_text = " ".join(sentences[start:end])
        chunks.append({
            "id": f"semantic_{i}",
            "text": chunk_text,
            "strategy": "semantic",
            "word_count": len(chunk_text.split()),
            "sentence_count": end - start
        })
    return chunks

# ── STRATEGY 4: AGENTIC ─────────────────────────────────────────────
def agentic_chunking(text: str) -> list:
    import json
    prompt = f"""You are a document chunking expert.
Analyze this text and identify where topics change.
Return ONLY a JSON object like this:
{{"splits": [
    {{"after_sentence": 3, "reason": "Topic changes from CPU to storage"}},
    {{"after_sentence": 7, "reason": "New section about connectivity"}}
]}}

TEXT:
{text[:3000]}

Rules:
- Identify 2-5 natural topic boundaries
- Return ONLY valid JSON, nothing else"""

    print("  Calling Groq LLM for agentic chunking...")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(raw)
        splits = result.get("splits", [])
        print(f"  LLM identified {len(splits)} split points")
        for s in splits:
            print(f"  -> After sentence {s['after_sentence']}: {s['reason']}")
    except json.JSONDecodeError:
        print(f"  LLM returned invalid JSON, using fallback")
        splits = []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    split_indices = sorted([s["after_sentence"] for s in splits])
    split_indices = [0] + split_indices + [len(sentences)]
    reasons = {s["after_sentence"]: s["reason"] for s in splits}

    chunks = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        chunk_text = " ".join(sentences[start:end])
        chunks.append({
            "id": f"agentic_{i}",
            "text": chunk_text,
            "strategy": "agentic",
            "word_count": len(chunk_text.split()),
            "split_reason": reasons.get(end, "document end")
        })
    return chunks

# ── COMPARE ALL 4 ───────────────────────────────────────────────────
def compare_all_strategies():
    print("=" * 60)
    print("COMPARING ALL 4 CHUNKING STRATEGIES")
    print("=" * 60)

    text = extract_text()
    print(f"\nSource: {len(text)} chars, {len(text.split())} words\n")

    strategies = {
        "1. Fixed Size": lambda: fixed_size_chunking(text),
        "2. Recursive":  lambda: recursive_chunking(text),
        "3. Semantic":   lambda: semantic_chunking(text),
        "4. Agentic":    lambda: agentic_chunking(text),
    }

    for name, fn in strategies.items():
        print(f"\n{'─'*40}")
        print(f"Running: {name}")
        chunks = fn()
        word_counts = [c["word_count"] for c in chunks]
        print(f"  Total chunks : {len(chunks)}")
        print(f"  Min words    : {min(word_counts)}")
        print(f"  Max words    : {max(word_counts)}")
        print(f"  Avg words    : {sum(word_counts)//len(word_counts)}")
        for c in chunks:
            print(f"  [{c['id']}] {c['word_count']} words -> {c['text'][:80]}...")

if __name__ == "__main__":
    compare_all_strategies()
