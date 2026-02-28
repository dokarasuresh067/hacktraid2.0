from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Configure Ollama model
model = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# Store conversation history
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite only if there's history
    if chat_history:
        messages = [
            SystemMessage(content=(
                "You are a search query rewriter. "
                "Rewrite the question as a short 3-6 word search query. "
                "Output ONLY the rewritten query. Nothing else."
            )),
        ] + chat_history + [
            HumanMessage(content=f"Rewrite this question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        print(f"  Doc {i}: {'. '.join(lines)}...")

    if not docs:
        answer = "No relevant products found in the database."
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))
        print(f"\nAnswer: {answer}")
        return

    # Step 3: Build context prompt
    combined_input = f"""Answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Answer directly using only the documents above. If the product is found, give specific details like price. If not found, say "This product is not available in our store."
"""

    # Step 4: Answer WITHOUT chat history (context docs are enough)
    messages = [
        SystemMessage(content=(
            "You are a helpful product assistant. "
            "Answer using ONLY the provided documents. "
            "Be concise and specific."
        )),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Save to history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer: {answer}")
    return answer


def start_chat():
    print(f"Model: {model.model}")
    print("Ask me questions! Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()