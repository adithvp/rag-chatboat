import nltk
import numpy as np

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)


def dynamic_semantic_chunk(
    text: str,
    min_sentences: int = 2
):
    """
    Dynamic semantic chunking based on sentence similarity.
    """

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= min_sentences:
        return [text]

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    similarities = []

    for i in range(len(sentences) - 1):

        score = cosine_similarity(
            [embeddings[i]],
            [embeddings[i + 1]]
        )[0][0]

        similarities.append(score)

    similarities = np.array(similarities)

    threshold = (
        np.mean(similarities)
        - np.std(similarities)
    )

    chunks = []
    current_chunk = [sentences[0]]

    for idx, sim in enumerate(similarities):

        if (
            sim < threshold
            and len(current_chunk) >= min_sentences
        ):

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = [
                sentences[idx + 1]
            ]

        else:

            current_chunk.append(
                sentences[idx + 1]
            )

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks

def semantic_chunk_documents(
    documents,
    embeddings=None
):
    """
    Convert LangChain documents
    into semantically chunked documents.
    """

    all_chunks = []

    for doc in documents:

        chunk_texts = dynamic_semantic_chunk(
            doc.page_content
        )

        for chunk in chunk_texts:

            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return all_chunks