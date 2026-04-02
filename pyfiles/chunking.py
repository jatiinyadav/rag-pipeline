# Chunking Data

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# Download once
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_split_documents(
    documents,
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.75,
    min_chunk_size=2,
    max_chunk_size=10
):
    """
    Semantic chunking based on sentence similarity.

    Parameters:
    - similarity_threshold: lower → fewer splits, higher → more splits
    - min_chunk_size: minimum sentences per chunk
    - max_chunk_size: maximum sentences per chunk
    """

    model = SentenceTransformer(model_name, device="cpu")
    final_chunks = []

    for doc in documents:
        text = doc.page_content

        # Step 1: Sentence split
        sentences = sent_tokenize(text)

        if len(sentences) == 0:
            continue

        # Step 2: Generate embeddings
        embeddings = model.encode(sentences)

        # Step 3: Build chunks
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])

            # Decide split
            if (
                sim < similarity_threshold
                or len(current_chunk) >= max_chunk_size
            ):
                # Ensure min size
                if len(current_chunk) >= min_chunk_size:
                    final_chunks.append({
                        "content": " ".join(current_chunk),
                        "metadata": doc.metadata
                    })
                    current_chunk = []
            
            current_chunk.append(sentences[i])

        # Add last chunk
        if current_chunk:
            final_chunks.append({
                "content": " ".join(current_chunk),
                "metadata": doc.metadata
            })

    print(f"Semantic split into {len(final_chunks)} chunks")

    # Example output
    if final_chunks:
        print("\nExample chunk:")
        print(final_chunks[0]["content"][:200])
        print("Metadata:", final_chunks[0]["metadata"])

    return final_chunks


def recursive_split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """
    Split documents into smaller chunks for better RAG performance.
    
    Parameters:
    - chunk_size: Maximum characters per chunk (adjust based on your LLM)
    - chunk_overlap: Characters to overlap between chunks (preserves context)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Each chunk: ~1000 characters
        chunk_overlap=chunk_overlap, # 200 chars overlap for context
        length_function=len, # How to measure length
        separators=["\n\n", "\n", " ", ""] # Split hierarchy
    )
    # Actually split the documents
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show what a chunk looks like
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs
