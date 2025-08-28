from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain imports for RAG
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
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

# Example: your website URL
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
    # Get the last user message
    query = state["messages"][-1]["content"]
    # Retrieve and generate response
    response = rag_chain.run(query)
    return {"messages": [{"role": "assistant", "content": response}]}

# -------------------------
# 7. Build graph
# -------------------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# -------------------------
# 8. Ready to use
# -------------------------
print("RAG-enabled LangGraph chatbot ready!")
