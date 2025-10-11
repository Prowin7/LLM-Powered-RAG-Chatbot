import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- LangChain Core Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def main():
    """Run the main Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon="📚")

    # --- Session State Setup ---
    # This is like the app's short-term memory
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- UI Header ---
    st.header("📚 Chat with your PDFs")
    st.write("Upload your documents and ask questions!")

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
                    
                    # --- Step 1: Read all PDFs into one big text string ---
                    raw_text = ""
                    for pdf in pdf_files:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            raw_text += page.extract_text() or ""
                    
                    # --- Step 2: Split the text into smaller chunks ---
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    text_chunks = text_splitter.split_text(raw_text)

                    # --- Step 3: Create the knowledge base (Vector Store) ---
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

                    # --- Step 4: Create the conversation "brain" ---
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                    )
                    
                    st.success("✅ Documents are ready! You can now ask questions.")

    # --- Main Chat Interface ---
    if st.session_state.conversation:
        user_question = st.text_input("💬 Ask a question about your documents:")
        if user_question:
            # Get the bot's response
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]

    # --- Display Chat History ---
    if st.session_state.chat_history:
        st.write("---")
        for i, message in enumerate(st.session_state.chat_history):
            # If the index is even, it's a user message
            if i % 2 == 0:
                st.write(f"**🧑 You:** {message.content}")
            # If the index is odd, it's a bot message
            else:
                st.write(f"**🤖 Bot:** {message.content}")

if __name__ == "__main__":
    main()

