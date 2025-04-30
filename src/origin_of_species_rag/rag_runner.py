from models.rag_agent import RagAgent
from models.pre_retrieval_agent import PreRetrievalAgent
from langchain.schema import Document
from core.database.vector_store import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class RagRunner:
    vector_store: VectorStore

    def __init__(self, vector_store: VectorStore):
        print("Starting RagRunner...")
        self.rag_agent = RagAgent()
        self.pre_retrieval_agent = PreRetrievalAgent()
        self.vector_store = vector_store
        print("RagRunner started")

    def get_rag_chain(self):
        return (
            { "context": lambda x: x["documents"], "question": lambda x: x["original_query"]} |
            self.rag_agent.get_chain()
        )

    def get_full_rag_chain(self, rag_chain):
        return (
            RunnablePassthrough() | 
            {
                "pre_retrieval_output": self.pre_retrieval_agent.get_chain(),
                "original_query": RunnablePassthrough()
            } |
            RunnableLambda(self.run_documents_and_original_query)
            | rag_chain
            | StrOutputParser()
        )
    
    async def run(self, user_query: str) -> str:
        """
        The run method is the main method that will be called to run the RAG.
        It will generate a response to a user's query based on the context provided by the vector database and do all the pre-processing and post-processing steps.
        """
        # This chain is used to generate a response to a user's query
        # based on the context provided by the vector database.
        rag_chain = self.get_rag_chain()

        # This is the full RAG chain.
        # It first generates a pre-retrieval query, then retrieves the context,
        # and finally generates a response to the user's query.
        full_rag_chain = self.get_full_rag_chain(rag_chain)

        return await full_rag_chain.ainvoke(user_query)
    
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

        print(f'The candidates are: {str(candidates)}')

        # Retrieve the relevant documents based on the original query
        relevant_documents = await self.vector_store.retrieve_similar_documents(original_query)
        final_documents: list[Document] = []
        for document in relevant_documents:
            final_documents.extend(document)
            final_documents.extend("\n\n")

        # Retrieve the relevant documents based on the candidates
        for candidate in candidates:
            final_documents.extend("\n\n")
            relevant_documents = await self.vector_store.retrieve_similar_documents(candidate)
            for document in relevant_documents:
                final_documents.extend(document)
                final_documents.extend("\n\n")

        # Return all documents in the end
        return final_documents
