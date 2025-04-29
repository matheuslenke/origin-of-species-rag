from typing import AsyncIterator, Iterator

from dotenv import load_dotenv
from models.rag_agent import RagAgent
from models.pre_retrieval_agent import PreRetrievalAgent
from langchain.schema import Document
from core.database.vector_store import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from asyncio import run as async_run
import nltk

class RagRunner:
    vector_store: VectorStore

    def __init__(self):
        self.rag_agent = RagAgent()
        self.pre_retrieval_agent = PreRetrievalAgent()
        self.vector_store = self.prepare_vector_store()

    async def prepare_vector_store() -> VectorStore:
        # Load the environment variables
        load_dotenv(verbose=True, override=True)

        # Prepare the NLTK tokenizer
        nltk.download('punkt_tab')

        vector_store = VectorStore()
        await vector_store.initialize()
        return vector_store

    async def run(self, user_query: str) -> AsyncIterator[str]:
        # This chain is used to generate a response to a user's query
        # based on the context provided by the vector database.
        rag_chain = (
            { "context": lambda x: x["documents"], "question": lambda x: x["original_query"]} |
            self.rag_agent.get_chain()
        )

        # This is the full RAG chain.
        # It first generates a pre-retrieval query, then retrieves the context,
        # and finally generates a response to the user's query.
        full_rag_chain = (
            RunnablePassthrough() | 
            {
                "pre_retrieval_output": self.pre_retrieval_agent.get_chain(),
                "original_query": RunnablePassthrough()
            } |
            RunnableLambda(self.run_documents_and_original_query)
            | rag_chain
        )
        async for chunk in full_rag_chain.astream(user_query):
            yield chunk
    
    async def run_documents_and_original_query(self, input_dict: dict) -> list[Document]:
        pre_retrieval_output = input_dict["pre_retrieval_output"]
        original_query = input_dict["original_query"]
        context_string = await self.generate_context(pre_retrieval_output, original_query)
        return {"documents": context_string, "original_query": original_query}

    async def generate_context(self, pre_retrieval_output: str, original_query: str) -> str:
        """
        Takes the output of the pre-retrieval step and the original query to fetch context
        and returns it as a single formatted string.
        """
        candidates = [item.strip() for item in pre_retrieval_output.split(',') if item.strip()]
        if not candidates and not original_query:
             return "No query or candidates provided."

        print(candidates)

        relevant_documents = await self.vector_store.retrieve_similar_documents(original_query)
        final_documents: list[Document] = []
        for document in relevant_documents:
            final_documents.extend(document)
            final_documents.extend("\n\n")

        for candidate in candidates:
            final_documents.extend("\n\n")
            relevant_documents = await self.vector_store.retrieve_similar_documents(candidate)
            for document in relevant_documents:
                final_documents.extend(document)
                final_documents.extend("\n\n")
        return final_documents
