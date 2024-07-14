import re
from typing import List
from llama_index.core.schema import TransformComponent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import Field

class CustomTransformation(TransformComponent):
    """
    A custom transformation component that applies a series of text preprocessing steps to the nodes.
    """
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = node.text.lower()
            node.text = re.sub(r'\s+', ' ', node.text)  # Replace multiple spaces with a single space
            node.text = re.sub(r'[^\w\s]', '', node.text)  # Removes punctuation
        return nodes

# class EmbedModel(TransformComponent):
#     """
#     A transformation component that applies an embedding model to the nodes.
#     """
#     embedding_model: object = Field(default=None, exclude=True)

#     def __init__(self, **data):
#         super().__init__(**data)
#         self.embedding_model = HuggingFaceEmbedding(
#             model_name="BAAI/bge-small-en-v1.5",
#             embed_batch_size=100
#         )

#     def __call__(self, nodes: List[object]) -> List[object]:
#         for node in nodes:
#             node.embedding = self.embedding_model.get_text_embedding(node.text)
#         return nodes

def Sentence_Splitter_docs_into_nodes(all_documents):
    """
    Splits the documents into nodes using a sentence splitter.
    """
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
    try:
        # Load data from directory
        documents = SimpleDirectoryReader(input_dir=r"C:\Users\pavan\Desktop\Generative AI\RAG-Using-Hybrid-Search-and-Re-Ranker\data").load_data()
        print(f"Loaded {len(documents)} documents")

        if documents:
            # Apply custom transformation
            custom_transform = CustomTransformation()
            documents = custom_transform(documents)

            # Split documents into nodes
            nodes = Sentence_Splitter_docs_into_nodes(documents)

            # Initialize embedding model
            #embed_model = EmbedModel()

            # Apply embedding model
            #nodes = embed_model(nodes)

            print(f"Created {len(nodes)} nodes")

            # Check the embedding for the first node
            #print(nodes[0].embedding)

        else:
            print("No documents to process.")

    except Exception as e:
        print(f"Error processing documents: {e}")
