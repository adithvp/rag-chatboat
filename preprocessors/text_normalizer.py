import re
import unicodedata
from langchain_core.documents import Document

def normalize_text(text: str) -> str:
    """
    Normalize text before chunking.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated newlines
    text = re.sub(r"\n+", "\n", text)

    return text.strip()



def normalize_documents(documents):
    """
    Normalize LangChain documents.
    """

    normalized_docs = []

    for doc in documents:

        normalized_text = normalize_text(
            doc.page_content
        )

        normalized_docs.append(
            Document(
                page_content=normalized_text,
                metadata=doc.metadata
            )
        )

    return normalized_docs