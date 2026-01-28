import pandas as pd
from langchain_core.documents import Document


def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        documents.append(
            Document(
                page_content=text,
                metadata={"source": row.get("source", "MedQuAD")}
            )
        )
    return documents
