from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage

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
    Extract the last user message (HumanMessage or dict),
    then retrieve and generate RAG response.
    """
    last_msg = state["messages"][-1]

    # Support LangChain message objects
    if hasattr(last_msg, "content"):
        query = last_msg.content
    else:
        query = last_msg["content"]

    response = rag_chain.run(query)

    # Always return as a list of dict messages for LangGraph
    return {"messages": [{"role": "assistant", "content": response}]}

# -------------------------
# 7. Build graph
# -------------------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

print("✅ RAG-enabled LangGraph chatbot ready!")

