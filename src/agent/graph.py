from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from bs4 import BeautifulSoup
import requests

# -------------------------
# 1. Define the state type
# -------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# -------------------------
# 2. Load website content safely
# -------------------------
def load_website_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # raises HTTPError for bad responses
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text
    except Exception as e:
        print(f"⚠️ Could not load website {url}: {e}")
        return ""  # fallback to OpenAI

website_text = load_website_text("https://emerico.com")

# -------------------------
# 3. Split text into chunks
# -------------------------
if website_text:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(website_text)

    # -------------------------
    # 4. Setup embeddings & vector store
    # -------------------------
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # -------------------------
    # 5. Create RAG chain
    # -------------------------
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever
    )
else:
    rag_chain = None  # RAG not available

# -------------------------
# 6. Initialize LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------
# 7. Define chatbot node
# -------------------------
def chatbot(state: State):
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]

    # Build chat history for RAG
    chat_history = []
    for msg in state["messages"][:-1]:
        if hasattr(msg, "content"):
            role = getattr(msg, "role", "user")
            if role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            else:
                chat_history.append(AIMessage(content=msg.content))
        else:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

    answer = None

    # 1️⃣ Try RAG if available
    if rag_chain:
        try:
            rag_response = rag_chain.run({
                "question": query,
                "chat_history": chat_history
            })
            if rag_response.strip():
                answer = rag_response
        except Exception as e:
            print(f"⚠️ RAG chain failed: {e}")

    # 2️⃣ Fallback to plain OpenAI if RAG returned nothing
    if not answer:
        try:
            response = llm([HumanMessage(content=query)])
            answer = response[0].content
        except Exception as e:
            print(f"⚠️ OpenAI fallback failed: {e}")
            answer = "Sorry, I could not generate a response at this time."

    return {"messages": [{"role": "assistant", "content": answer}]}

# -------------------------
# 8. Build graph
# -------------------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

print("✅ Fully working RAG + OpenAI fallback LangGraph chatbot ready!")
