import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.prompts.system import system_prompt, human_message
from langchain_core.messages import BaseMessage
from typing import List
from langchain_core.output_parsers import StrOutputParser


class RagAgent:
    """
    The RAG agent is responsible for generating a response to a user's query
    based on the context provided by the vector database.
    """

    chat_history: List[BaseMessage] = []

    def __init__(self):
        gemini_api_key = str(os.environ["GEMINI_API_KEY"])
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=gemini_api_key
        )

    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])
        return prompt | self.model | StrOutputParser()

    def generate_query(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.invoke(prompt_with_user_query)

    def generate_query_stream(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.stream(prompt_with_user_query)
