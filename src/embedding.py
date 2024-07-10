import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embedding_model(embedding_model_name, embed_batch_size):
    embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            embed_batch_size=embed_batch_size,
        )
    return embedding_model

class EmbedModel():
    def __init__(self):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size= 100,
        )

    def __call__(self, node):
        embeddings = self.embedding_model.get_text_embedding(node["text"])
        return embeddings
