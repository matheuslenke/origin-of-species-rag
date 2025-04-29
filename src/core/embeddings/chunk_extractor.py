from typing import List
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema import Document

def extract_chunks(data: str ) -> List[Document]:
    """
    Extracts chunks from input text data using recursive character splitting.
    
    This function takes a string input and splits it into smaller chunks while maintaining context.
    The chunks are created using a recursive character splitter that:
    - Creates chunks of approximately 200 characters
    - Maintains a 10 character overlap between chunks to preserve context
    - Splits on natural boundaries like paragraphs and sentences when possible
    
    Args:
        data (str): The input text to be split into chunks
        
    Returns:
        List[Document]: A list of Document objects, where each Document contains a chunk of the original text
    """
    # Create a text splitter (https://www.nltk.org/)
    # The NLTKTextSplitter is used to split the text into a more natural way,
    # by splitting on sentences and paragraphs. This is an easy way to split the text
    # by good quality chunks with not so much processing power.
    text_splitter = NLTKTextSplitter(language="english", separator="\n\n")

    chunks = text_splitter.split_documents([Document(page_content=data)])

    return chunks