from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.prompts.system import system_prompt, human_message
from langchain_core.messages import BaseMessage
from typing import List

class RetrievalAgent:
    """
    The retrieval agent is responsible for retrieving the most relevant chunks of a document
    based on a user's query.

    It uses a Google gemini 2.0 flash model, which is a fast and efficient model
    for this kind of task.
    """
    chat_history: List[BaseMessage] = []

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def generate_query(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.invoke(prompt_with_user_query)
    
    def generate_query_stream(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.stream(prompt_with_user_query)
