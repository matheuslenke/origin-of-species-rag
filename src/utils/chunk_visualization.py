from typing import List, Tuple
from langchain.schema import Document
import os


def visualize_chunks_markdown(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """
    Generates a Markdown table visualization for a list of documents and their scores.

    Args:
        docs_with_scores: A list of tuples, where each tuple contains a
                          Langchain Document and its associated score (float).

    Returns:
        A Markdown string representing the table visualization.
    """
    markdown_content = """
| Score | Content |
|-------|---------|
"""

    for doc, score in docs_with_scores:
        # Escape pipe characters within the content to avoid breaking Markdown table formatting
        escaped_content = doc.page_content.replace("|", "\|").replace("\n", " ")
        markdown_content += f"| {score:.4f} | {escaped_content} |\n"

    save_markdown_file(markdown_content)
    return markdown_content


def save_markdown_file(markdown_content: str, filename: str = "chunk_visualization.md"):
    """Saves the markdown content to a file in the data directory."""
    data_dir = "data"
    # Ensure the data directory exists (though user confirmed, it's good practice)
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "w") as f:
        f.write(markdown_content)
    print(f"Generated {filepath}")
