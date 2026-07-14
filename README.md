# Agentic RAG: Smart AI Assistant 🤖

This repository contains an advanced **Agentic Retrieval-Augmented Generation (RAG)** application. It is designed to act as a smart assistant that dynamically decides whether to retrieve answers from an uploaded PDF document (using a Vector Database) or fetch the latest information from the internet (using a Web Search API).

## 🌟 Key Features
- **Intelligent Routing**: The agent understands the context of your question and intelligently routes it to the correct data source (Vectorstore or Web Search).
- **Local AI Embeddings**: Uses HuggingFace's local sentence-transformer models to embed PDF data securely and free of cost, avoiding API limits.
- **Dynamic Web Search**: Capable of querying the internet for real-time events or general knowledge that isn't present in the uploaded documents.
- **Beautiful UI**: An aesthetically pleasing, dark-themed Streamlit interface.

## 🛠️ Technology Stack
- **Framework**: [LangChain](https://python.langchain.com/) & [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) (for the agentic workflow)
- **Large Language Model (LLM)**: OpenAI API (via [OpenRouter](https://openrouter.ai/)) for fast and powerful text generation.
- **Embeddings Model**: `all-MiniLM-L6-v2` via HuggingFace (Local processing)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Local persistent storage for document chunks)
- **Web Search API**: [Tavily Search](https://tavily.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)

## 🚀 How It Works
1. **Document Upload**: You upload a PDF document. The app splits it into chunks and embeds it locally using HuggingFace.
2. **Query Routing**: When you ask a question, the LLM evaluates the prompt. 
    - If the query is related to the PDF, it routes to the `retrieve` node.
    - If the query requires current knowledge, it routes to the `web_search` node.
3. **Answer Generation**: The context (either from the PDF or the Web) is passed back to the LLM to generate a comprehensive, accurate response.

## 💻 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd agentic_rag_project
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENROUTER_API_KEY="your_openrouter_key"
   TAVILY_API_KEY="your_tavily_key"
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```
