from dotenv import load_dotenv
load_dotenv()

from src.build_vectorstore import build_vectorstore
from src.rag_pipeline import create_rag_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def main():
    VECTORSTORE_PATH = "vectorstore"
    DATASET_PATH = "data/medquad.csv"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(VECTORSTORE_PATH):
        print("Building vector store from dataset...")
        vectorstore = build_vectorstore(DATASET_PATH)
    else:
        print("Loading existing vector store...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Initializing RAG pipeline...")
    rag_chain = create_rag_chain(vectorstore)

    print("\n✅ RAG system ready. Type 'exit' to quit.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        answer = rag_chain.invoke(query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
