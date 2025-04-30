system_prompt = {
    "role": "system",
    "content": """You are an AI assistant designed to optimize user queries for retrieval.
        Your task is to take chunks of a text that were retrieved from a vector database and
        summarize and organize them in a way that other AI models can understand and use to answer
        the user's question.
        """
}

human_message = {
    "role": "user",
    "content": """
        <user_query> {user_query} </user_query>
        <chunks> {chunks} </chunks>
    """
}
