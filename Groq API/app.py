import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Import the Document class

from dotenv import load_dotenv

load_dotenv()

## Load the GROQ and OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq With Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    """Initialize vector embeddings and store them in session state."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        
        # Load data from a CSV file
        st.session_state.csv_file_path = "./dollar.csv"  # Change to your CSV file path
        df = pd.read_csv(st.session_state.csv_file_path)

        # Convert DataFrame to a list of LangChain documents
        st.session_state.docs = []
        for _, row in df.iterrows():
            text = ' '.join(row.astype(str))  # Convert row to string
            # Create a Document object
            st.session_state.docs.append(Document(page_content=text))

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Create vector embeddings


prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

# Ensure vector store is ready before proceeding
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Set k=3 to retrieve top 3 most relevant documents
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})  # Set k=3
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time: ", time.process_time() - start)
    
    # Display the response
    st.write(response['answer'])

    # With a Streamlit expander to show similar documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)  # This now works since doc is a Document object
            st.write("--------------------------------")
else:
    if prompt1:
        st.write("Please initialize the vector store by clicking 'Documents Embedding' first.")
