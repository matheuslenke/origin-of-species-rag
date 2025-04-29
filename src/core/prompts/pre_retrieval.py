from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate

system_prompt = SystemMessage("""You are an AI assistant designed to optimize user queries for retrieval.
        Your task is to take a user's query and transform it into a more effective search query for a vector database.
        The goal is to maximize the relevance of the retrieved results by focusing on the most important aspects of the user's question.
        Please return 3 possible queries, with the most relevant first.
        """)

human_message = PromptTemplate(
    """
    User Query: {user_query}
    """,
    role="user",
    input_variables=["user_query"]
)