"""
Merged retriever across Chroma collections:
- faq     : FAQ entries (1 doc per entry)
- tickets : resolved support tickets (1 doc per ticket)
- guides  : PDF guide chunks (split at ingest)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "Chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.8   # lower threshold for better recall

# ── Build Retriever ──────────────────────────────────────────────────────────
def build_retriever(k_faq: int = 3, k_tickets: int = 3, k_guides: int = 3) -> RunnableLambda:
    """Creates a merged retriever across FAQ, tickets, and guides collections."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Initialize stores
    faq_store = Chroma(collection_name="faq", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    tickets_store = Chroma(collection_name="tickets", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    guides_store = Chroma(collection_name="guides", embedding_function=embeddings, persist_directory=CHROMA_DIR)

    def retrieve(query: str):
        # Query each collection
        faq_results = faq_store.similarity_search_with_score(query, k=k_faq)
        ticket_results = tickets_store.similarity_search_with_score(query, k=k_tickets)
        guide_results = guides_store.similarity_search_with_score(query, k=k_guides)

        # Merge results
        all_results = faq_results + ticket_results + guide_results
        if not all_results:
            return {"fallback": True, "docs": []}

        # Sort by similarity score (lower = better)
        all_results.sort(key=lambda x: x[1])
        best_score = all_results[0][1]
        print(f"Best similarity score: {best_score:.3f}")

        # Apply threshold
        if best_score > SIMILARITY_THRESHOLD:
            print("⚠️ Threshold triggered — fallback mode")
            return {"fallback": True, "docs": []}

        # Return docs only
        docs = [doc for doc, score in all_results]
        return {"fallback": False, "docs": docs}

    return RunnableLambda(retrieve)

# ── Quick Test ───────────────────────────────────────────────────────────────
def get_retrieval_result(query: str):
    retriever = build_retriever()
    return retriever.invoke(query)
