import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- LangChain Core Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatHuggingFace

# Define FAISS index file path
FAISS_DB_PATH = "./faiss_index"

def main():
    """Run the main Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon="📚")

    # --- Session State Setup ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- UI Header ---
    st.header("📚 Chat with your PDFs")
    st.write("Upload your documents, process them, and start chatting!")

    # --- Load Hugging Face API Key ---
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_key:
        st.error("❌ Please set your Hugging Face Hub API key in the .env file as HUGGINGFACEHUB_API_TOKEN")
        st.stop()

    # --- Load Existing FAISS DB On Startup ---
    if os.path.exists(FAISS_DB_PATH) and st.session_state.conversation is None:
        with st.spinner("🔄 Loading existing FAISS knowledge base..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            
            llm = ChatHuggingFace(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=hf_api_key)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            st.sidebar.success("✅ Knowledge base loaded from FAISS!", icon="💽")

    # --- Sidebar for File Uploads ---
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_files = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if not pdf_files:
                st.warning("⚠️ Please upload at least one PDF.")
            else:
                with st.spinner("🔄 Reading and preparing your documents..."):
                    # Step 1: Read PDFs
                    raw_text = ""
                    for pdf in pdf_files:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text() or ""
                    
                    # Step 2: Split text into chunks
                    text_splitter = CharacterTextSplitter(
                        separator="\n", chunk_size=1000,
                        chunk_overlap=200, length_function=len
                    )
                    text_chunks = text_splitter.split_text(raw_text)

                    # Step 3: Create FAISS vector store and save
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
                    vectorstore.save_local(FAISS_DB_PATH)

                    # Step 4: Create conversation chain
                    llm = ChatHuggingFace(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=hf_api_key)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                    )
                    
                    st.success("✅ Documents processed and saved to FAISS!")

    # --- Main Chat Interface ---
    if st.session_state.conversation:
        user_question = st.text_input("💬 Ask a question about your documents:")
        if user_question:
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]

    # --- Display Chat History ---
    if st.session_state.chat_history:
        st.write("---")
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**🧑 You:** {message.content}")
            else:
                st.write(f"**🤖 Bot:** {message.content}")

if __name__ == "__main__":
    main()
