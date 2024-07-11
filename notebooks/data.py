import re
from llama_index.core.schema import TransformComponent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pydantic import BaseModel, Field
from typing import List
# Initialize the embedding model
#huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#embed_model = LangchainEmbedding(huggingface_embeddings)

# Define a custom transformation component
class CustomTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = node.text.lower()
            node.text = re.sub(r'\s+', ' ', node.text)  # Replace multiple spaces with a single space
            node.text = re.sub(r'[^\w\s]', '', node.text)  # Removes punctuation
        return nodes

# # Define the embedding model transformation component
# class EmbeddingModel(TransformComponent):
#     def __init__(self):
#         self.model = embed_model

#     def __call__(self, nodes):
#         for node in nodes:
#             node.embedding = self.model.get_text_embedding(node.text)
#         return nodes
    
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embedding_model(embedding_model_name, embed_batch_size):
    embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            embed_batch_size=embed_batch_size
        )
    return embedding_model

class EmbedModel(TransformComponent):
    embedding_model: object = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.embedding_model = get_embedding_model(
            embedding_model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=100
        )

    def __call__(self, nodes: List[object]) -> List[object]:
        for node in nodes:
            node.embedding = self.embedding_model.get_text_embedding(node.text)
        return nodes

# # Create the ingestion pipeline
# pipeline = IngestionPipeline(
#     transformations=[
#         CustomTransformation(),
#         SentenceSplitter(chunk_size=1024, chunk_overlap=20),
#         EmbedModel(),
#     ]
# )

# if __name__ == '__main__':
#     # Load data from directory
    
#         reader = SimpleDirectoryReader(input_dir=r"C:\Users\pavan\Desktop\Generative AI\RAG-Using-Hybrid-Search-and-Re-Ranker\data\RAG_PDF")
#         documents = reader.load_data()
#         print(f"Loaded {len(documents)} documents")
        
#         # Run the ingestion pipeline
#         nodes = pipeline.run(show_progress= True, documents=documents)
#         print(f"Created {len(nodes)} nodes")
    
def Sentence_Splitter_docs_into_nodes(all_documents):
    try:
        splitter = SentenceSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        nodes = splitter.get_nodes_from_documents(all_documents)

        return nodes

    except Exception as e:
        print(f"Error splitting documents into nodes: {e}")
        return []
    
if __name__ == '__main__':
    # Load data from directory

        #reader = SimpleDirectoryReader(input_dir=r"C:\Users\pavan\Desktop\Generative AI\RAG-Using-Hybrid-Search-and-Re-Ranker\data")
        documents = SimpleDirectoryReader(input_dir=r"C:\Users\pavan\Desktop\Generative AI\RAG-Using-Hybrid-Search-and-Re-Ranker\data").load_data(show_progress = True)
        print(f"Loaded {len(documents)} documents")
        if documents:
            documents = CustomTransformation(documents)

            # Split documents into nodes
            nodes = Sentence_Splitter_docs_into_nodes(documents)

            # Initialize embedding model
            embeddings = EmbedModel(nodes)
        else:
            print("No documents to process.")

        # Run the ingestion pipeline
        #nodes_parsed = pipeline.run(documents=documents)
        print(f"Created {len(embeddings)} nodes")
        