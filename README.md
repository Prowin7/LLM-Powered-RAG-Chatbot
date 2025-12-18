# RAG - ChatBot: Retrieval Augmented Generation (RAG) chatbot using HuggingFace's Zephyr-7B-Beta, LangChain, FAISS, and Streamlit

This RAG-ChatBot is a Python application that allows users to chat with PDF documents. It uses HuggingFace's Zephyr-7B-Beta model to generate accurate answers based on the content of uploaded PDFs, with responses grounded in the document content.

## Key Features

- **Document Processing**: Upload and process PDF documents on the fly
- **Persistent Storage**: FAISS vector database maintains document embeddings between sessions
- **Contextual Understanding**: Maintains conversation history for coherent multi-turn discussions
- **Source Verification**: Tracks document sources for response verification
- **Efficient Search**: Uses FAISS for fast similarity search across document chunks

## Technical Stack

- **Language Model**: HuggingFace Zephyr-7B-Beta
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain for RAG pipeline
- **Web Interface**: Streamlit
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

## How It Works

1. **Document Ingestion**:
   - PDFs are processed to extract text
   - Text is split into manageable chunks
   - Chunks are converted to vector embeddings using HuggingFace embeddings
   - Vectors are stored in a FAISS index

2. **Query Processing**:
   - User questions are converted to embeddings
   - FAISS retrieves the most relevant document chunks
   - The language model generates responses using the retrieved context

3. **Conversation Management**:
   - Maintains chat history for context-aware responses
   - Uses LangChain's ConversationBufferMemory
   - Implements ConversationalRetrievalChain for coherent dialog

## Setup Instructions

### Prerequisites
- Python 3.8+
- HuggingFace API token

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd chatbot
