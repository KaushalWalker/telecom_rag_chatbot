
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "Chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



test_cases = [
    ("My internet is not working at all", "TK-001"),
    ("Network signal keeps dropping", "TK-002"),
    ("My data usage is not updating", "TK-003"),
    ("I was charged while roaming unexpectedly", "TK-004"),
    ("My SIM stopped working after software update", "TK-005"),
    ("I got billed twice this month", "TK-006"),
    ("People can't call me and it goes to voicemail", "TK-007"),
    ("My 4G speed is extremely slow", "TK-008"),
    ("Unable to activate my eSIM", "TK-009"),
    ("Number porting is taking too long", "TK-018"),
]

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

tickets_store = Chroma(
    collection_name="tickets",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

hits = 0

for query, expected_ticket in test_cases:

    results = tickets_store.similarity_search(
        query,
        k=3
    )

    retrieved_ids = [
        doc.metadata["ticket_id"]
        for doc in results
    ]

    print(f"\nQuery: {query}")
    print(f"Expected: {expected_ticket}")
    print(f"Retrieved: {retrieved_ids}")

    if expected_ticket in retrieved_ids:
        hits += 1

recall = hits / len(test_cases)

print(f"\nRecall@3 = {recall:.2%}")