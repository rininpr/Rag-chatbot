import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

# Paths
PDF_PATH = os.getenv("PDF_PATH", str(BASE / "documents" / "Introducing Mac Studio 2025 (en_GB).pdf"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE / "chromadb"))

# Models
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("mac_studio")

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

from chunkers import chunk_text  # noqa: E402  (pure, unit-tested)

def ingest():
    print(f"Reading PDF: {PDF_PATH}")
    text = extract_text(PDF_PATH)
    print(f"Total characters: {len(text)}")

    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings and storing in ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.upsert(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": "Mac_Studio_2025", "chunk_index": i}]
        )
        if i % 10 == 0:
            print(f"  Processed {i}/{len(chunks)} chunks...")

    print(f"Done! {len(chunks)} chunks stored in ChromaDB")
    print(f"Collection count: {collection.count()}")

if __name__ == "__main__":
    ingest()
