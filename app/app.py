import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- LangChain Core Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma  # Changed from FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Define the path for the persistent ChromaDB
CHROMA_DB_PATH = "./chroma_db"

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

    # --- Load Existing DB On Startup ---
    # Smartly load the conversation chain if the DB already exists
    if os.path.exists(CHROMA_DB_PATH) and st.session_state.conversation is None:
        with st.spinner("🔄 Loading existing knowledge base..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            st.sidebar.success("✅ Knowledge base loaded from disk!", icon="💽")


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

                    # Step 3: Create ChromaDB vector store and persist it
                    embeddings = OpenAIEmbeddings()
                    vectorstore = Chroma.from_texts(
                        texts=text_chunks, 
                        embedding=embeddings,
                        persist_directory=CHROMA_DB_PATH  # This saves the DB
                    )

                    # Step 4: Create conversation chain
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                    )
                    
                    st.success("✅ Documents processed and saved!")

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

