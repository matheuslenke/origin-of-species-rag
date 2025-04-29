from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.prompts.pre_retrieval import system_prompt, human_message

class PreRetrievalAgent:
    """
    The pre-retrieval agent is responsible for getting the raw user query and try
    to optimize it for the retrieval process, considering that the question might
    not be semantically close to the correct answer in a vector database.

    It uses a Google gemini 2.0 flash lite model, which is a fast and efficient model
    for this kind of task.
    """
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

    def generate_query(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.invoke(prompt_with_user_query)
    
    def generate_query_stream(self, user_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

        prompt_with_user_query = prompt.format_prompt(user_query=user_query)
        return self.model.stream(prompt_with_user_query)
