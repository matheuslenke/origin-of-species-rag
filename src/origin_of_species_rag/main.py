import asyncio
from typing import List
from dotenv import load_dotenv
import streamlit as st
import nltk

from core.embeddings.file_embeddings import create_file_embeddings
from core.database.vector_store import VectorStore
from langchain.schema import Document

from utils.chunk_visualization import visualize_chunks_markdown

mockup_query = "What is the struggle for existence?"

async def prepare_vector_store() -> VectorStore:
    # Load the environment variables
    load_dotenv(verbose=True, override=True)

    # Prepare the NLTK tokenizer
    nltk.download('punkt_tab')

    vector_store = VectorStore()
    await vector_store.initialize()
    return vector_store


async def transform_book_and_save_embeddings(vector_store: VectorStore):
    documents: List[Document] = await create_file_embeddings("data/the_origin_of_species.txt")

    vector_store.save_embeddings(documents=documents)
    
    saved_documents = vector_store.retrieve_all_documents()
    print(f'{len(saved_documents)} documents retrieved from vector store')

def chat_loop():
    while True:
        user_query = input("Enter a query: ")
        print(user_query)


async def main():
    print("Starting Origin of Species RAG application...")
    try:
        vector_store = await prepare_vector_store()

        # Ideally this function should be called only once, when you are preparing the database
        # for production use.
        # await transform_book_and_save_embeddings(vector_store)

        results = await vector_store.retrieve_similar_documents(mockup_query)
        visualize_chunks_markdown(results)

    except Exception as e:
        print("An exception ocurred on application initialization")
        # st.dialog("An exception ocurred on application initialization. Please restart the application.")
        # st.error("An exception ocurred on application initialization. Please restart the application.")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
