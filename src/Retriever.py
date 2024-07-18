import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from typing import List, Union
from rerank import reranking

# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hybrid_search():
    """
    A class for performing hybrid search using dense and sparse embeddings.
    """

    def __init__(self) -> None:
        """
        Initialize the Hybrid_search object with dense and sparse embedding models and a Qdrant client.
        """
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

    def metadata_filter(self, file_names: Union[str, List[str]]) -> models.Filter:
        """
        Create a metadata filter based on the file names provided.

        Args:
            file_names (Union[str, List[str]]): A single file name or a list of file names.

        Returns:
            models.Filter: A Qdrant filter object based on the file names provided.
        """
        if isinstance(file_names, str):
            
            file_name_condition = models.FieldCondition(
                key="file_name",
                match=models.MatchValue(value=file_names)
            )
        else:
            
            file_name_condition = models.FieldCondition(
                key="file_name",
                match=models.MatchAny(any=file_names)
            )

        return models.Filter(
            must=[file_name_condition]
        )

    def query_hybrid_search(self, query, metadata_filter=None, limit=5):
        """
        Perform a hybrid search using dense and sparse embeddings.

        Args:
            query (str): The query string.
            metadata_filter (models.Filter, optional): A Qdrant filter object based on metadata. Defaults to None.
            limit (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            List[models.ScoredPoint]: A list of scored points based on the query and metadata filter.
        """
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
            query_filter=metadata_filter,
            query=models.FusionQuery(fusion=models.Fusion.RRF), #Reciprocal Rerank Fusion
        )
        
        # Extract the text from the payload of each scored point
        documents = [point.payload['text'] for point in results.points]

        return documents

if __name__ == '__main__':
    search = Hybrid_search()
    query = "Can you explain what is Adaptive Retrieval?"
    file_names = "Adaptive-RAG.pdf"
    metadata_filter = search.metadata_filter(file_names)
    results = search.query_hybrid_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for query: {query}")
    #print(results)
    
    # # Rerank the documents using a reranking model
    reranked_documents = reranking.rerank_documents(query, results)
    print(reranked_documents)