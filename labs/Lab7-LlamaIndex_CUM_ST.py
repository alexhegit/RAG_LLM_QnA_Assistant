import os
# Enalbe 780M with ROCm
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import streamlit as st
#import openai
#from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Set embedding model
# Please download it ahead running this lab by "ollama pull nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Set ollama model
Settings.llm = Ollama(model="llama3", request_timeout=200.0)

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        service_context = ServiceContext.from_defaults(llm=Settings.llm, 
                                                           embed_model=Settings.embed_model, 
                                                           system_prompt="You are an expert on the Car User Manual and your job is to answer technical questions. Assume that all questions are related to the User Car Manual. Keep your answers technical and based on facts – do not hallucinate features.")
        
        if not os.path.exists("./chroma_db/CUM_en_db"):
            # initialize client
            db = chromadb.PersistentClient(path="./chroma_db/CUM_en_db")
            # get collection
            chroma_collection = db.get_or_create_collection("CUM_en_db")
            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Load data
            docs = SimpleDirectoryReader(input_files=["../data/FordUM.pdf"]).load_data()
            # Build vector index per-document
            index = VectorStoreIndex.from_documents(
                docs,
                service_context=service_context, 
                storage_context=storage_context,
                transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=50)],
            )
        else:
            # initialize client
            db = chromadb.PersistentClient(path="./chroma_db/CUM_en_db")
            # get collection
            chroma_collection = db.get_or_create_collection("CUM_en_db")
            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # load your index from stored vectors
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context, 
                storage_context=storage_context
            )
 
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history