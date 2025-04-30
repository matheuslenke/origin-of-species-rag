from dotenv import load_dotenv
from core.database.vector_store import VectorStore


async def prepare_vector_store() -> VectorStore:
    print("Starting Origin of Species RAG application...")
    # Load the environment variables
    load_dotenv(verbose=True, override=True)

    # Prepare the NLTK tokenizer
    # Only need to download once
    # nltk.download('punkt_tab', './nltk_data')

    vector_store = VectorStore()
    await vector_store.initialize()
    print("Vector store prepared")
    return vector_store
