from core.database.vector_store import VectorStore
from langchain_core.documents import Document
from typing import List
from time import sleep
from core.embeddings.file_embeddings import create_file_embeddings


async def transform_file_and_save_embeddings(vector_store: VectorStore):
    """
    This function is used to transform the entire file into smaller chunks
    and save the embeddings to the vector store. It is not necessary to run
    that function every time you run the application, only when you are
    preparing the database for the first time.
    """
    documents: List[Document] = await create_file_embeddings(
        "data/the_origin_of_species.txt"
    )

    for doc in documents:
        vector_store.save_embeddings(documents=[doc])
        sleep(
            5
        )  # The sleep is necessary due to the rate limit of the Gemini API free tier.

    saved_documents = vector_store.retrieve_all_documents()
    print(f"{len(saved_documents)} documents retrieved from vector store")
