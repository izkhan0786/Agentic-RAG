import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# LangGraph Imports
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Setup Streamlit Page
st.set_page_config(page_title="Agentic RAG | Financial Analyst", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0E1117;}
    h1 {color: #00E5FF; font-family: 'Inter', sans-serif;}
    .stChatFloatingInputContainer {padding-bottom: 20px;}
    .css-1d391kg {background-color: #1E2127;} /* Sidebar color */
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic RAG: Smart AI Assistant")
st.markdown("Upload a PDF document. The Agent will dynamically decide whether to answer from the **PDF** (Vectorstore) or **Search the Web** (Tavily) for the latest information.")

# Check for API Keys
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not openrouter_api_key:
    st.warning("⚠️ **OPENROUTER_API_KEY** is missing. Please add it to your `.env` file.")
if not tavily_key or tavily_key == "your_tavily_api_key_here":
    st.warning("⚠️ **TAVILY_API_KEY** is missing. Please add it to your `.env` file.")

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF (e.g., Financial Report)", type=["pdf"])

# Application State for LangGraph
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    keys: dict

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF
if uploaded_file and not st.session_state.vectorstore:
    with st.spinner("Processing PDF and creating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=50
            )
            doc_splits = text_splitter.split_documents(docs)
            
            if openrouter_api_key:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    collection_name="rag-chroma",
                    embedding=embeddings,
                )
                st.success("✅ PDF embedded successfully! You can now ask questions about it.")
            else:
                st.error("Cannot embed PDF without an OpenRouter API Key.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LangGraph Logic Functions ---
def retrieve(state):
    st.toast("Agent chose: **Vectorstore Retrieval**", icon="📚")
    state_dict = state["keys"]
    question = state_dict["question"]
    retriever = st.session_state.vectorstore.as_retriever()
    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "question": question}}

def web_search(state):
    st.toast("Agent chose: **Web Search**", icon="🌐")
    state_dict = state["keys"]
    question = state_dict["question"]
    
    if tavily_key and tavily_key != "your_tavily_api_key_here":
        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\\n\\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
    else:
        web_results = Document(page_content="Web search unavailable without Tavily API Key.")
    
    return {"keys": {"documents": [web_results], "question": question}}

def generate(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key, model="openai/gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        template="""You are an expert AI assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:""",
        input_variables=["question", "context"],
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    return {"keys": {"documents": documents, "question": question, "generation": generation}}

def route_question(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    
    # If no PDF is uploaded, always route to web search
    if not st.session_state.vectorstore:
        return "web_search"
    
    # Use LLM to route
    llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key, model="openai/gpt-4o-mini", temperature=0)
    system_prompt = """You are an expert routing assistant. Your job is to decide whether to route a user's question to a vector database or a web search tool.
    The vector database contains documents uploaded by the user (e.g., Financial Reports, PDFs).
    If the question is about the uploaded document, output 'vectorstore'.
    If the question is about recent news, current events, or general knowledge not in the document, output 'web_search'.
    Output ONLY the exact word 'vectorstore' or 'web_search'. No other text."""
    
    prompt = PromptTemplate(
        template=system_prompt + "\\n\\nQuestion: {question}\\nRouting Decision:",
        input_variables=["question"],
    )
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": question})
    
    if "web_search" in decision.lower():
        return "web_search"
    return "vectorstore"

# --- Build LangGraph ---
app_graph = None
if openrouter_api_key:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    app_graph = workflow.compile()

# Chat Input
if prompt := st.chat_input("Ask a question about the PDF or recent news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    if app_graph:
        with st.chat_message("assistant"):
            with st.spinner("Agent is deciding the best way to answer..."):
                inputs = {"keys": {"question": prompt}}
                for output in app_graph.stream(inputs):
                    pass # Streamlit toasts handle the progress updates
                
                # Get the final generation from the last node (generate)
                final_state = output[list(output.keys())[0]]
                final_generation = final_state["keys"]["generation"]
                
                st.markdown(final_generation)
                st.session_state.messages.append({"role": "assistant", "content": final_generation})
    else:
        st.error("Please provide an OpenRouter API key in `.env` to start the agent.")
