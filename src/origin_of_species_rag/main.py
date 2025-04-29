import asyncio
from origin_of_species_rag.rag_runner import RagRunner

async def chat_loop(rag_runner: RagRunner):
    while True:
        user_query = input("Enter a query: ")
        print(user_query)
        async for chunk in rag_runner.run(user_query=user_query):
            print(chunk, end="")
        print()

async def main():
    print("Starting Origin of Species RAG application...")
    try:
        rag_runner = RagRunner()

        # This function should be called only once you are preparing the database
        # await transform_file_and_save_embeddings(vector_store)

        # Executes the main chat loop
        await chat_loop(rag_runner)

    except Exception as e:
        print("An exception ocurred on application initialization")
        # st.dialog("An exception ocurred on application initialization. Please restart the application.")
        # st.error("An exception ocurred on application initialization. Please restart the application.")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
