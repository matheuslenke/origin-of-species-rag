import streamlit as st
from .prompts.system import system_prompt

def initialize_streamlit():
    """
    Initializes the Streamlit application interface.
    
    This function sets up the main components of the Streamlit chat interface including:
    - Page configuration (title, icon, layout)
    - Sidebar with usage guide
    - Chat message history initialization
    - Message display system
    - Chat input field
    
    Returns:
        str: The user's input from the chat input field
    """
    
    st.set_page_config(
        page_title="Chat application with Rag - Origin of Species",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📖 Rag Origin of species - Matheus Lenke")
    with st.sidebar:
        st.header("📚 Guia de utilização deste ChatBot")
        st.markdown(""" 
            ## Como usar 🧐

            Bem-vindo ao ChatBot sobre "A Origem das Espécies" de Charles Darwin! 🐵➡️👨‍🔬
                    
            Desenvolvido por Matheus Lenke para o desafio de RAG da Boticário.

            1.  **Faça sua pergunta:** Digite qualquer dúvida que você tenha sobre o livro no campo de texto na parte inferior da tela. ⌨️
            2.  **Receba a resposta:** O ChatBot buscará informações diretamente no livro para fornecer a resposta mais precisa. 🧠
            3.  **Explore:** Continue a conversa fazendo mais perguntas ou refinando as anteriores. 🤓

            **Exemplo:** "Qual a principal ideia de Darwin sobre seleção natural?" 🤔

            Divirta-se explorando a obra prima de Darwin! 🎉
        """)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            system_prompt,
            { "role": "assistant", "content": "Vamos começar a conversar! 👇" }
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Skip system messages. They do not need to be shown for the user
        if (message["role"] != "system"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # create the bar where we can type messages
    prompt = st.chat_input("Por favor, faça uma pergunta.")
    return prompt