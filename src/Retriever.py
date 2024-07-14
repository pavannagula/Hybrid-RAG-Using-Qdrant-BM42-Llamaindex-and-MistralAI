import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector


Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

class Hybrid_search():

    def __init__(self) -> None:
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
                            url=Qdrant_URL,
                            api_key=Qdrant_API_KEY)
        
    def create_metadata_filter(self, file_names):
        filter_conditions = []
        for file_name in file_names:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.file_name",
                    match=models.MatchValue(value=file_name),
                ),
            )
        metadata_filter = models.Filter(should=filter_conditions)
        return metadata_filter
            
    def query_hybrid_search(self, query, metadata_filter=None, limit=10):
        # Embed the query using the dense embedding model
        dense_query = list(self.embedding_model.embed([query]))[0].tolist()

        # Embed the query using the sparse embedding model
        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

        results = self.qdrant_client.query_points(
            collection_name=Collection_Name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                    using="sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            filter=metadata_filter,
        )
        return results


if __name__ == '__main__':
    search = Hybrid_search()
    query = "Can you explain what is Adaptive Retrieval?"
    file_names = ["Adaptive-RAG.pdf", "RAFT.pdf", "Ragnarök.pdf","SELF-RAG.pdf"]
    metadata_filter = search.create_metadata_filter(file_names)
    results = search.query_hybrid_search(query, metadata_filter=metadata_filter)