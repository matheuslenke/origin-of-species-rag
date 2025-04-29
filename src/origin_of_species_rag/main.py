import asyncio
from time import sleep
from typing import List
from dotenv import load_dotenv
import nltk


from core.database.vector_store import VectorStore
from langchain.schema import Document

from origin_of_species_rag.rag_runner import RagRunner
from utils import transform_file_and_save_embeddings

async def prepare_vector_store() -> VectorStore:
    # Load the environment variables
    load_dotenv(verbose=True, override=True)

    # Prepare the NLTK tokenizer
    nltk.download('punkt_tab')

    vector_store = VectorStore()
    await vector_store.initialize()
    return vector_store



def chat_loop():
    while True:
        user_query = input("Enter a query: ")
        print(user_query)

async def main():
    print("Starting Origin of Species RAG application...")
    try:
        vector_store = await prepare_vector_store()

        # This function should be called only once you are preparing the database
        # await transform_file_and_save_embeddings(vector_store)

        rag_runner = RagRunner(vector_store)
        mockup_query = "What is the struggle for existence?"
        runner = rag_runner.run(mockup_query)
        # print(str(runner))
        async for chunk in runner:
            print(chunk, end="")
        print() # Add a final newline

    except Exception as e:
        print("An exception ocurred on application initialization")
        # st.dialog("An exception ocurred on application initialization. Please restart the application.")
        # st.error("An exception ocurred on application initialization. Please restart the application.")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
