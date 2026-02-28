import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def row_to_text(row: pd.Series) -> str:
    """Convert a CSV row into a natural language sentence for better embedding quality."""

    def fmt_bool(val):
        return "Yes" if str(val).strip().lower() in ("true", "1", "yes") else "No"

    def fmt_price(val):
        try:
            return f"₹{float(val):,.2f}"
        except Exception:
            return str(val)

    return (
        f"Product: {row.get('product_name', 'N/A')} (ID: {row.get('product_id', 'N/A')}). "
        f"Category: {row.get('category', 'N/A')}, Brand: {row.get('brand', 'N/A')}. "
        f"Seller: {row.get('seller', 'N/A')} located in {row.get('seller_city', 'N/A')}. "
        f"Original price: {fmt_price(row.get('price', 0))}, "
        f"Discount: {row.get('discount_percent', 0)}%, "
        f"Final price: {fmt_price(row.get('final_price', 0))}. "
        f"Rating: {row.get('rating', 'N/A')} out of 5 based on {row.get('review_count', 0)} reviews. "
        f"Stock available: {row.get('stock_available', 0)} units, Units sold: {row.get('units_sold', 0)}. "
        f"Delivery in {row.get('delivery_days', 'N/A')} days. "
        f"Returnable: {fmt_bool(row.get('is_returnable', False))}, "
        f"Return window: {row.get('return_policy_days', 0)} days. "
        f"Warranty: {row.get('warranty_months', 0)} months. "
        f"Color: {row.get('color', 'N/A')}, Size: {row.get('size', 'N/A')}. "
        f"Weight: {row.get('weight_g', 'N/A')}g. "
        f"Payment modes accepted: {row.get('payment_modes', 'N/A')}. "
        f"Seller rating: {row.get('seller_rating', 'N/A')}. "
        f"Listed on: {row.get('listing_date', 'N/A')}."
    )


def load_csv_as_documents(docs_path: str = "docs") -> list[Document]:
    """
    Load all CSV files from docs_path, convert each row to a natural language
    Document, and attach the original row data as metadata for filtering.
    """
    print(f"Loading CSV files from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. "
            "Please create it and add your CSV files."
        )

    csv_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(docs_path)
        for f in files
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found in '{docs_path}'. Please add your CSV documents."
        )

    all_documents: list[Document] = []

    for csv_file in csv_files:
        print(f"  Processing: {csv_file}")
        df = pd.read_csv(csv_file, encoding="utf-8")
        df.columns = df.columns.str.strip()          # remove accidental whitespace
        df = df.where(pd.notna(df), other="N/A")     # replace NaN with readable string

        for _, row in df.iterrows():
            text = row_to_text(row)

            # Store every column as metadata so you can do Chroma filtered searches
            metadata = {
                "source": csv_file,
                **{
                    col: (
                        str(row[col])           # Chroma only accepts str/int/float/bool
                        if not isinstance(row[col], (int, float, bool))
                        else row[col]
                    )
                    for col in df.columns
                },
            }

            all_documents.append(Document(page_content=text, metadata=metadata))

    print(f"  ✅ Loaded {len(all_documents)} product documents from {len(csv_files)} file(s).\n")

    # Preview first 2 documents
    for i, doc in enumerate(all_documents[:2]):
        print(f"--- Document {i + 1} preview ---")
        print(doc.page_content)
        print(f"Metadata keys: {list(doc.metadata.keys())}\n")

    return all_documents


def create_vector_store(
    documents: list[Document],
    persist_directory: str = "db/chroma_db",
) -> Chroma:
    """Embed documents with BGE-small and persist to ChromaDB."""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},   # required for BGE cosine similarity
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"✅ Vector store saved to '{persist_directory}' "
          f"with {vectorstore._collection.count()} documents.\n")
    return vectorstore


def load_existing_vector_store(persist_directory: str = "db/chroma_db") -> Chroma:
    """Load an already-persisted ChromaDB collection."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"✅ Loaded existing vector store with "
          f"{vectorstore._collection.count()} documents.")
    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
        print("Vector store already exists. No need to re-process documents.")
        return load_existing_vector_store(persistent_directory)

    print("Persistent directory not found. Building vector store from scratch...\n")

    documents = load_csv_as_documents(docs_path)
    vectorstore = create_vector_store(documents, persistent_directory)

    print("✅ Ingestion complete! Your documents are ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()