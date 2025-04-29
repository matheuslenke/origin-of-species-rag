import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores import VectorStore
import chromadb
from langchain.schema import Document

class VectorStore:
    """
    Embeddings class for the Origin of Species RAG application.
    """
    embeddings: GoogleGenerativeAIEmbeddings
    client: chromadb.Client
    vector_store: Chroma

    def embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
    
    def save_embeddings(self, documents: List[Document]):
        self.vector_store.add_documents(documents=documents)
    
    async def initialize(self):
        self.client = self._initialize_chroma_client()
        self.embeddings = self._initialize_embeedings_model()
        self.vector_store = await self._initialize_vector_db()

    def _initialize_embeedings_model(self):
        print("Initializing Google Generative AI Embeddings Model...")
        gemini_api_key = str(os.environ["GEMINI_API_KEY"])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", google_api_key=gemini_api_key)
        print("VertexAI Embeddings Model initialized")
        return embeddings
    
    async def _initialize_vector_db(self) -> VectorStore:
        """
        Initializes and returns a vector database using Chroma with Ollama embeddings.
        
        This function sets up a vector store that can be used to store and retrieve 
        document embeddings. It uses the nomic-embed-text model from Ollama for 
        generating embeddings and Chroma as the vector database.
        
        Returns:
            VectorStore: A configured Chroma vector store instance ready for use
        """
        print("Initializing Chroma Vector Store...")

        vector_store = Chroma(
            collection_name="origin_of_species",
            embedding_function=self.embeddings,
            client=self.client
        )
        print("Chroma Vector Store initialized")
        return vector_store

    def _initialize_chroma_client(self):
        print("Initializing Chroma Client...")
        chroma_host = os.environ["CHROMA_DB_HOST"]
        chroma_port: int = int(os.environ["CHROMA_DB_PORT"])
        environment = os.environ["ENVIRONMENT"]
        ssl_option = environment == "production"

        client = chromadb.PersistentClient(path="./chroma_db")

        # is_server_running = client.heartbeat()

        # print(client.count_collections())

        # if not is_server_running:
        #     raise Exception("Chroma server is not running")
        # else:
        #     print("Chroma server is running on port", chroma_port)

        # collection_exists = await client.get_or_create_collection("origin_of_species")
        # print("Collection exists:", await collection_exists.count())
        return client


    async def retrieve_similar_documents(self, user_prompt: str):
        # creating and invoking the retriever

        docs: list[tuple[Document, float]] = await self.vector_store.asimilarity_search_with_score(user_prompt, k = 20)

        # docs = retriever.invoke(user_prompt)
        for doc, score in docs:
            print(f"Score: {score}")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 20)  # Separator for readability
        return docs

    def retrieve_all_documents(self):
        return self.vector_store.get()

