from typing import List
from core.embeddings import chunk_extractor
from langchain.schema import Document

async def create_file_embeddings(filename: str) -> List[Document]:
    with open(file=filename, mode="r") as file:
        data = file.read()
        chunks = chunk_extractor.extract_chunks(data)
    print(f'{len(chunks)} chunks extracted from {filename}')
    return chunks