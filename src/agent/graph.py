from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

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

# Example website
website_docs = [load_website_text("https://emerico.com")]

# -------------------------
# 3. Setup embeddings & vector store
# -------------------------
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(website_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 4. Initialize LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------
# 5. Create RAG chain
# -------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)

# -------------------------
# 6. Define chatbot node
# -------------------------
def chatbot(state: State):
    """
    Extract last user message, prepare chat history,
    then retrieve & generate RAG response.
    """
    last_msg = state["messages"][-1]

    # Get content from message object or dict
    query = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]

    # Build chat_history for ConversationalRetrievalChain
    chat_history = []
    for msg in state["messages"][:-1]:  # exclude current query
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

    # Run RAG chain
    response = rag_chain.run({
        "question": query,
        "chat_history": chat_history
    })

    # Return as LangGraph message list
    return {"messages": [{"role": "assistant", "content": response}]}

# -------------------------
# 7. Build graph
# -------------------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

print("✅ Full RAG-enabled LangGraph chatbot ready!")
