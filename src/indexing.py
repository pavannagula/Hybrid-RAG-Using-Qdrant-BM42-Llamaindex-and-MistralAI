import logging
from dotenv import load_dotenv
import os
import json
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm import tqdm

# Load environmental variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

class QdrantIndexing:
    """
    A class for indexing documents using Qdrant vector database.
    """

    def __init__(self) -> None:
        """
        Initialize the QdrantIndexing object.
        """
        self.data_path = r"\data\nodes.json"
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
                            url=Qdrant_URL,
                            api_key=Qdrant_API_KEY)
        self.metadata = []
        self.documents = []
        logging.info("QdrantIndexing object initialized.")

    def load_nodes(self, input_file):
        """
        Load nodes from a JSON file and extract metadata and documents.

        Args:
            input_file (str): The path to the JSON file.
        """
        with open(input_file, 'r') as file:
            self.nodes = json.load(file)

        for node in self.nodes:
            self.metadata.append(node['metadata'])
            self.documents.append(node['text'])

        logging.info(f"Loaded {len(self.nodes)} nodes from JSON file.")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if not self.qdrant_client.collection_exists(collection_name=f"{Collection_Name}"): 
            self.qdrant_client.create_collection(
                collection_name= Collection_Name,
                vectors_config={
                     'dense': models.VectorParams(
                         size=384,
                         distance = models.Distance.COSINE,
                     )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                              index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            logging.info(f"Created collection '{Collection_Name}' in Qdrant vector database.")

    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using SPLADE.
        """
        # Generate the sparse vector using SPLADE model
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]

        # Check if embeddings has indices and values attributes
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")


    def documents_insertion(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            # Generate both dense and sparse embeddings
            dense_embedding = list(self.embedding_model.embed([doc]))[0]
            sparse_vector = self.create_sparse_vector(doc)

            # Create PointStruct
            point = models.PointStruct(
                id=i,
                vector={
                    'dense': dense_embedding.tolist(),
                    'sparse': sparse_vector,
                },
                payload={
                    'text': doc,
                    **metadata  # Include all metadata
                }
            )
            points.append(point)

        # Upsert points
        self.qdrant_client.upsert(
            collection_name=Collection_Name,
            points=points
        )

        logging.info(f"Upserted {len(points)} points with dense and sparse vectors into Qdrant vector database.")

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    indexing = QdrantIndexing()
    indexing.load_nodes(indexing.data_path)
    indexing.client_collection()
    indexing.documents_insertion()
