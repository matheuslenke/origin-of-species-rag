import asyncio
from core.streamlit import initialize_streamlit
from origin_of_species_rag.rag_runner import RagRunner
import streamlit as st
from utils.prepare_vector_store import prepare_vector_store

async def chat_loop(rag_runner: RagRunner):
    user_prompt = initialize_streamlit()

    if user_prompt == "exit":
        return
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta..."):
                final_response = await rag_runner.run(user_query=user_prompt)
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

async def main():
    try:
        vector_store = await prepare_vector_store()
        
        rag_runner = RagRunner(vector_store)
        
        # This function should be called only once you are preparing the database
        # await transform_file_and_save_embeddings(vector_store)

        await chat_loop(rag_runner)

    except Exception as e:
        print("An exception ocurred on Chat application. Please restart the application.")
        st.error("An exception ocurred on Chat application. Please restart the application.")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
