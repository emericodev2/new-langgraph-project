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
# 2. Load website content
# -------------------------
def load_website_text(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    return text

website_text = load_website_text("https://emerico.com")

# -------------------------
# 3. Split text into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_text(website_text)

# -------------------------
# 4. Setup embeddings & vector store
# -------------------------
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 5. Initialize LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------
# 6. Create RAG chain
# -------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)

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

    # 1️⃣ Run RAG chain
    rag_response = rag_chain.run({
        "question": query,
        "chat_history": chat_history
    })

    # 2️⃣ Fallback to OpenAI if RAG found nothing
    if not rag_response.strip():
        response = llm.invoke([HumanMessage(content=query)])
        answer = response.content if hasattr(response, "content") else str(response)
    else:
        answer = rag_response

    return {"messages": [{"role": "assistant", "content": answer}]}

# -------------------------
# 8. Build graph
# -------------------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

print("✅ RAG + OpenAI fallback LangGraph chatbot ready!")
