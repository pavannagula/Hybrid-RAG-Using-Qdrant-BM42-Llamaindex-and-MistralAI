from sentence_transformers import CrossEncoder
from typing import List, Tuple
from qdrant_client.http.models import QueryResponse

class reranking():
    def __init__(self) -> None:
        pass

    def rerank_documents(self, query: str, results: QueryResponse, k: int = 3) -> List[str]:
        # Extract the text of the documents from the results object
        documents = [result.payload['text'] for result in results.points]

        # Load a cross-encoder model
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Create a list of query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Compute the scores for each pair
        scores = model.predict(pairs)

        # Combine the scores with the document text
        scored_documents = [(doc, score) for doc, score in zip(documents, scores)]

        # Sort the documents by score in descending order
        sorted_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

        # Return the top-scoring documents
        return [doc for doc, score in sorted_documents[:k]]
