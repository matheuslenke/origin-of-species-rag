from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

system_prompt = SystemMessage("""You are an AI assistant that will answer questions about a specific document.
        You will receive chunks of the document and a question.
        You will answer the question based on the context provided.
        """)

human_message = ChatPromptTemplate("""
        Question: {question}
        Context: {context}
                              
        Answer:
        """,
        role="user",
        input_variables=["question", "context"])