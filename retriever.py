"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda

SIMILARITY_THRESHOLD = 1.2
CHROMA_DIR = "Chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    def retrieve(query: str):

        faq_results = faq_store.similarity_search_with_score(
            query,
            k=k_faq
        )

        ticket_results = tickets_store.similarity_search_with_score(
            query,
            k=k_tickets
        )

        guide_results = guides_store.similarity_search_with_score(
            query,
            k=k_guides
        )

        all_results = (
            faq_results +
            ticket_results +
            guide_results
        )

        if not all_results:
            return {
                "fallback": True,
                "docs": []
            }

        all_results.sort(key=lambda x: x[1])

        best_score = all_results[0][1]

        print(f"Best similarity score: {best_score}")

        if best_score > SIMILARITY_THRESHOLD:
            print("THRESHOLD TRIGGERED")

            return {
                "fallback": True,
                "docs": []
            }

        return {
            "fallback": False,
            "docs": [doc for doc, score in all_results]
        }

    return RunnableLambda(retrieve)



def get_retrieval_result(query: str):

    retriever = build_retriever()

    return retriever.invoke(query)