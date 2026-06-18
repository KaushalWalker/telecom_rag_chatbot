"""
CLI entry point for the telecom RAG chatbot.
Usage: python main.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from rag_chain import build_chain
from retriever import get_retrieval_result

load_dotenv()


def main():
    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    chain = build_chain()

    while True:
        question = input("Customer: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        result = get_retrieval_result(question)

        if result["fallback"]:
            print("\nAssistant: I don't know. Please call 611.\n")
            continue
        
        # print("CALLING LLM...")
        print("\nAssistant: ", end="", flush=True)

        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    main()