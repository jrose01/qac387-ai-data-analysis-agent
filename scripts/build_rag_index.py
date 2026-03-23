from __future__ import annotations

import argparse
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_faiss_utils_pdf import (
    build_faiss_index,
    build_rag_chunks,
    save_faiss_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save a FAISS RAG index from markdown files."
    )
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default="knowledge",
        help="Directory containing markdown files for the RAG corpus.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use when creating vectors.",
    )
    args = parser.parse_args()

    knowledge_dir = Path(args.knowledge_dir)
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    print(f"Reading markdown files from: {knowledge_dir}")
    chunks = build_rag_chunks(knowledge_dir)

    if not chunks:
        raise ValueError(
            "No markdown chunks were created. Make sure your knowledge folder contains .md files."
        )

    print(f"Built {len(chunks)} chunks.")
    print(f"Creating embeddings with model: {args.embedding_model}")

    index, _ = build_faiss_index(
        chunks=chunks,
        embedding_model=args.embedding_model,
    )

    index_path, meta_path = save_faiss_index(
        knowledge_dir=knowledge_dir,
        index=index,
        chunks=chunks,
        embedding_model=args.embedding_model,
    )

    print("\nRAG index build complete.")
    print(f"Saved FAISS index   : {index_path}")
    print(f"Saved chunk metadata: {meta_path}")
    print("\nNext step:")
    print(
        "Load these files inside your agent at startup, then retrieve top-k chunks before code generation."
    )


if __name__ == "__main__":
    main()
