import os
import sys
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Page configuration
st.set_page_config(
    page_title="RAG Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        color: #e0e4e8;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(22, 33, 62, 0.95);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #4ade80;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-indicator-yellow {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #fbbf24;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Info cards */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Upload section styling */
    .upload-section {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: inline-block;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def save_uploaded_files(uploaded_files, temp_dir):
    """Save uploaded files to temporary directory."""
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
        print(f"[DEBUG] Saved file: {file_path}")
    print(f"[DEBUG] Total files saved: {len(saved_files)}")
    return saved_files


@st.cache_resource
def get_rag_search(_persist_dir, embedding_model, llm_model):
    """Get cached RAG search instance for a specific persist directory."""
    return RAGSearch(
        persist_dir=_persist_dir,
        embedding_model=embedding_model,
        llm_model=llm_model
    )


def initialize_rag_from_documents(documents, persist_dir, embedding_model, llm_model):
    """Initialize RAG system from uploaded documents."""
    from langchain_groq import ChatGroq
    
    # Create vector store and build from documents
    vectorstore = FaissVectorStore(persist_dir, embedding_model)
    vectorstore.build_from_documents(documents)
    
    # Create RAG search instance
    rag_search = RAGSearch(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        llm_model=llm_model
    )
    
    return rag_search


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 RAG Document Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your documents and get AI-powered answers</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    if "temp_persist_dir" not in st.session_state:
        st.session_state.temp_persist_dir = tempfile.mkdtemp(prefix="rag_")
    
    # Configuration from environment
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"**Embedding Model:** `{embedding_model}`")
        st.markdown(f"**LLM Model:** `{llm_model}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Number of results slider
        top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Reset documents button
        if st.button("🔄 Reset Documents", use_container_width=True):
            st.session_state.rag_initialized = False
            st.session_state.uploaded_file_names = []
            st.session_state.messages = []
            # Clean up temp directory
            if os.path.exists(st.session_state.temp_persist_dir):
                shutil.rmtree(st.session_state.temp_persist_dir, ignore_errors=True)
            st.session_state.temp_persist_dir = tempfile.mkdtemp(prefix="rag_")
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        if st.session_state.rag_initialized:
            st.markdown('<span class="status-indicator"></span> **System Ready**', unsafe_allow_html=True)
            st.markdown(f"📄 **{len(st.session_state.uploaded_file_names)}** documents loaded")
        else:
            st.markdown('<span class="status-indicator-yellow"></span> **Awaiting Documents**', unsafe_allow_html=True)
    
    # Main content area
    if not st.session_state.rag_initialized:
        # Document upload section
        st.markdown("### 📁 Upload Your Documents")
        st.markdown("*Supported formats: PDF, TXT, CSV, DOCX, XLSX, JSON*")
        
        uploaded_files = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
            for f in uploaded_files:
                st.markdown(f"- � {f.name} ({f.size / 1024:.1f} KB)")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Process Documents & Start Chat", use_container_width=True):
                    with st.spinner("📚 Processing documents..."):
                        try:
                            # Create temporary directory for uploaded files
                            temp_upload_dir = tempfile.mkdtemp(prefix="rag_upload_")
                            
                            # Save uploaded files
                            saved_files = save_uploaded_files(uploaded_files, temp_upload_dir)
                            st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                            
                            # Load documents
                            with st.status("Loading documents...", expanded=True) as status:
                                st.write(f"📖 Reading {len(saved_files)} uploaded files...")
                                for sf in saved_files:
                                    st.write(f"  - {os.path.basename(sf)}")
                                
                                documents = load_all_documents(temp_upload_dir)
                                st.write(f"✅ Loaded {len(documents)} document chunks from {len(saved_files)} files")
                                
                                if not documents:
                                    st.error("❌ No valid documents found. Please check your files.")
                                    st.stop()
                                
                                st.write("🔄 Building vector store...")
                                # Build vector store
                                vectorstore = FaissVectorStore(
                                    st.session_state.temp_persist_dir, 
                                    embedding_model
                                )
                                vectorstore.build_from_documents(documents)
                                st.write(f"✅ Vector store created with {len(documents)} chunks")
                                
                                st.write("🤖 Initializing RAG system...")
                                status.update(label=f"✅ Processed {len(saved_files)} files → {len(documents)} chunks!", state="complete")
                            
                            # Clean up temp upload directory
                            shutil.rmtree(temp_upload_dir, ignore_errors=True)
                            
                            st.session_state.rag_initialized = True
                            st.success("🎉 Documents processed! You can now start asking questions.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error processing documents: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            # Empty state
            st.markdown("""
            <div class="upload-section">
                <h3>📤 No documents uploaded yet</h3>
                <p>Upload your documents to get started with AI-powered search</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Chat interface (documents are loaded)
        # Header with reset option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 💬 Chat with Your Documents")
        with col2:
            if st.button("📤 Upload New Documents", use_container_width=True):
                # Reset everything
                st.session_state.rag_initialized = False
                st.session_state.uploaded_file_names = []
                st.session_state.messages = []
                if os.path.exists(st.session_state.temp_persist_dir):
                    shutil.rmtree(st.session_state.temp_persist_dir, ignore_errors=True)
                st.session_state.temp_persist_dir = tempfile.mkdtemp(prefix="rag_")
                st.cache_resource.clear()
                st.rerun()
        
        # Show loaded documents info with delete option
        with st.expander("📄 Loaded Documents", expanded=False):
            for fname in st.session_state.uploaded_file_names:
                st.markdown(f"- {fname}")
            st.markdown("---")
            st.caption("Click 'Upload New Documents' above to start fresh with different files.")
        
        # Initialize RAG system
        try:
            rag_search = RAGSearch(
                persist_dir=st.session_state.temp_persist_dir,
                embedding_model=embedding_model,
                llm_model=llm_model
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.stop()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents and generating response..."):
                    try:
                        response = rag_search.search_and_summarize(prompt, top_k=top_k)
                        st.markdown(response)
                        
                        # Add assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"❌ An error occurred: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
