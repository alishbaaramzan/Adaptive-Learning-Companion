"""
Lab 2: ingest_data.py
Adaptive Learning Companion — PDF Knowledge Base Ingestion Pipeline
─────────────────────────────────────────────────────────────────────
Reads a course PDF → cleans text → semantic chunking → metadata enrichment
→ embeds via OpenAI → stores in ChromaDB

Usage:
    python ingest_data.py --pdf your_book.pdf --topic "AI" --difficulty intermediate

Install:
    pip install pdfplumber chromadb openai python-dotenv
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple

import pdfplumber
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────
# STEP 1: EXTRACT TEXT FROM PDF
# ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text page-by-page using pdfplumber."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # Tag each page so we can track source page in metadata
                full_text += f"\n\n[PAGE_{page_num}]\n{text}"
    print(f"✓ Extracted {len(full_text):,} characters from {pdf_path}")
    return full_text


# ─────────────────────────────────────────────────────────
# STEP 2: CLEAN TEXT
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip domain-specific noise:
    - Page headers / footers / numbering
    - Copyright notices
    - HTML artifacts
    - URLs
    - Excessive whitespace / punctuation
    """
    # Page numbers (standalone digits or "Page X of Y")
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Copyright / legal boilerplate
    text = re.sub(r'Copyright\s+©?\s*\d{4}.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'All rights reserved.*', '', text, flags=re.IGNORECASE)

    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # URLs
    text = re.sub(r'https?://\S+', '', text)

    # Repeated dashes / underscores used as dividers
    text = re.sub(r'[-_]{4,}', '', text)

    # Normalize bullet symbols
    text = re.sub(r'^\s*[•●▪◦]\s*', '• ', text, flags=re.MULTILINE)

    # Collapse excessive punctuation
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# ─────────────────────────────────────────────────────────
# STEP 3: SEMANTIC CHUNKING
# ─────────────────────────────────────────────────────────

def semantic_chunk(text: str,
                   max_chunk_size: int = 1000,
                   overlap: int = 150) -> List[Dict]:
    """
    Chunk strategy: paragraph-aware with overlap.
    - Splits on double newlines (paragraph boundaries) to keep
      related sentences together — avoids cutting mid-concept.
    - Overlap carries the last N characters into next chunk so
      context is not lost at boundaries.
    - Tracks page number from [PAGE_N] markers.
    """
    chunks = []
    current_chunk = ""
    current_page = 1
    chunk_start_page = 1
    idx = 0

    # Split preserving page markers
    segments = re.split(r'(\[PAGE_\d+\])', text)

    for segment in segments:
        page_match = re.match(r'\[PAGE_(\d+)\]', segment)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        paragraphs = segment.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= max_chunk_size:
                if not current_chunk:
                    chunk_start_page = current_page
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': idx,
                        'start_page': chunk_start_page,
                    })
                    idx += 1
                # Overlap: carry last `overlap` chars into new chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + para + "\n\n"
                chunk_start_page = current_page

    # Final chunk
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'chunk_index': idx,
            'start_page': chunk_start_page,
        })

    print(f"✓ Created {len(chunks)} semantic chunks")
    return chunks


# ─────────────────────────────────────────────────────────
# STEP 4: METADATA ENRICHMENT
# ─────────────────────────────────────────────────────────

def build_metadata(chunk_text: str,
                   chunk_index: int,
                   start_page: int,
                   source_file: str,
                   topic: str,
                   difficulty: str) -> Dict[str, str]:
    """
    Attach at least 3 mandatory searchable tags + extra signals.

    MANDATORY (per Lab 2 spec):
        1. topic          — subject domain
        2. difficulty     — beginner | intermediate | advanced
        3. content_type   — auto-detected from text signals

    ADDITIONAL:
        4. source_file    — original PDF filename
        5. chunk_index    — position in document
        6. start_page     — PDF page number for citation
        7. char_count     — length for filtering
        8. last_updated   — ingestion timestamp
        9. doc_id         — unique MD5 for deduplication
       10. has_examples   — signals presence of illustrative content
       11. has_definitions— signals definitional content
       12. has_steps      — signals procedural/how-to content
    """
    # Auto-detect content type from text patterns
    if re.search(r'exercise|problem|question \d+|quiz|task \d+', chunk_text, re.I):
        content_type = "practice"
    elif re.search(r'prerequisite|before.*study|prior knowledge|required.*understand', chunk_text, re.I):
        content_type = "prerequisites"
    else:
        content_type = "explanation"

    doc_id = hashlib.md5(
        f"{source_file}_{chunk_index}_{chunk_text[:60]}".encode()
    ).hexdigest()

    return {
        # ── MANDATORY 3 ──────────────────────────────────
        "topic":        topic.lower().replace(" ", "_"),
        "difficulty":   difficulty,
        "content_type": content_type,
        # ── ADDITIONAL ───────────────────────────────────
        "source_file":  os.path.basename(source_file),
        "chunk_index":  str(chunk_index),
        "start_page":   str(start_page),
        "char_count":   str(len(chunk_text)),
        "last_updated": datetime.now().isoformat(),
        "doc_id":       doc_id,
        # ── CONTENT FEATURE FLAGS ─────────────────────────
        "has_examples":    str(bool(re.search(r'for example|e\.g\.|such as|consider', chunk_text, re.I))),
        "has_definitions": str(bool(re.search(r'is defined as|refers to|means that|is called', chunk_text, re.I))),
        "has_steps":       str(bool(re.search(r'step \d+|first[,\s]|second[,\s]|finally[,\s]', chunk_text, re.I))),
    }


# ─────────────────────────────────────────────────────────
# STEP 5: EMBED VIA OPENAI
# ─────────────────────────────────────────────────────────

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Batch embed texts using OpenAI embeddings API."""
    # ChromaDB can auto-embed, but explicit embeddings give us model control
    response = openai.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]
    print(f"✓ Generated {len(embeddings)} embeddings using {model}")
    return embeddings


# ─────────────────────────────────────────────────────────
# STEP 6: STORE IN CHROMADB
# ─────────────────────────────────────────────────────────

def store_in_chromadb(chunks: List[Dict],
                      embeddings: List[List[float]],
                      collection_name: str = "learning_companion_kb") -> chromadb.Collection:
    """
    Store chunks, embeddings, and metadata in a persistent ChromaDB collection.
    Uses a named collection scoped to this project (Lab 2 'namespace' requirement).
    """
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"project": "Adaptive Learning Companion", "lab": "Lab2"}
    )

    documents  = [c['text']     for c in chunks]
    metadatas  = [c['metadata'] for c in chunks]
    ids        = [c['metadata']['doc_id'] for c in chunks]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✓ Stored {len(documents)} chunks → collection '{collection_name}'")
    print(f"  Total documents in DB: {collection.count()}")
    return collection


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def ingest(pdf_path: str,
           topic: str,
           difficulty: str,
           collection_name: str = "learning_companion_kb"):

    print(f"\n{'='*60}")
    print("ADAPTIVE LEARNING COMPANION — INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"  PDF:        {pdf_path}")
    print(f"  Topic:      {topic}")
    print(f"  Difficulty: {difficulty}")
    print(f"{'='*60}\n")

    # 1. Extract
    raw_text = extract_text_from_pdf(pdf_path)

    # 2. Clean
    clean = clean_text(raw_text)

    # 3. Chunk
    chunks = semantic_chunk(clean)

    # 4. Enrich metadata
    for chunk in chunks:
        chunk['metadata'] = build_metadata(
            chunk_text=chunk['text'],
            chunk_index=chunk['chunk_index'],
            start_page=chunk['start_page'],
            source_file=pdf_path,
            topic=topic,
            difficulty=difficulty
        )

    # 5. Embed
    texts = [c['text'] for c in chunks]
    embeddings = get_embeddings(texts)

    # 6. Store
    collection = store_in_chromadb(chunks, embeddings, collection_name)

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"\nSample metadata (chunk 0):")
    print(json.dumps(chunks[0]['metadata'], indent=2))
    print(f"\nSample text (chunk 0 preview):")
    print(chunks[0]['text'][:300], "...\n")

    return collection


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ADAPTIVE LEARNING COMPANION — PDF INGESTION")
    print("="*60 + "\n")

    # PDF path
    while True:
        pdf_path = input("Path to your PDF file: ").strip()
        if os.path.exists(pdf_path):
            break
        print(f"  ❌ File not found: '{pdf_path}'. Please try again.")

    # Topic
    topic = input("Topic / subject name (e.g. 'machine_learning'): ").strip()
    if not topic:
        topic = "general"

    # Difficulty
    print("Difficulty level:")
    print("  1) beginner")
    print("  2) intermediate")
    print("  3) advanced")
    difficulty_map = {"1": "beginner", "2": "intermediate", "3": "advanced"}
    choice = input("Choose 1/2/3 (default: 2): ").strip()
    difficulty = difficulty_map.get(choice, "intermediate")

    # Collection name
    collection = input("ChromaDB collection name (press Enter for 'learning_companion_kb'): ").strip()
    if not collection:
        collection = "learning_companion_kb"

    ingest(pdf_path, topic, difficulty, collection)