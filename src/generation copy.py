from llama_index.core.query_pipeline import QueryPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import Dict, Any

def create_query_pipeline(
    qdrant_client,
    collection_name: str,
    embedding_dimension: int,
    top_k: int = 10,
    rerank_top_n: int = 5,
    metadata_filter: Dict[str, Any] = None
) -> QueryPipeline:
    # Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        dimension=embedding_dimension
    )

    # Set up retriever with metadata filter
    retriever = VectorIndexRetriever(
        vector_store=vector_store,
        similarity_top_k=top_k,
        filters=metadata_filter
    )

    # Set up reranker
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=rerank_top_n
    )

    # Set up LLM
    llm = OpenAI(model="gpt-3.5-turbo")

    # Set up prompt template
    prompt_template = PromptTemplate(
        "Given the context information: {context_str}, "
        "please answer the following question: {query_str}\n"
        "If the information is not present in the context, please say 'I don't have enough information to answer this question.'"
    )

    # Set up summarizer
    summarizer = get_response_synthesizer(
        response_mode="compact",
        use_async=True
    )

    # Create query pipeline
    pipeline = QueryPipeline(verbose=True)

    # Add modules to the pipeline
    pipeline.add_modules({
        "retriever": retriever,
        "reranker": reranker,
        "prompt": prompt_template,
        "llm": llm,
        "summarizer": summarizer
    })

    # Define links between modules
    pipeline.add_link("retriever", "reranker")
    pipeline.add_link("reranker", "prompt")
    pipeline.add_link("prompt", "llm")
    pipeline.add_link("llm", "summarizer")

    return pipeline

# Usage example
def query_pipeline(pipeline: QueryPipeline, query: str) -> str:
    response = pipeline.run(query_str=query)
    return str(response)

# Example of how to use the functions
if __name__ == "__main__":
    import qdrant_client

    # Initialize Qdrant client
    qdrant_client = qdrant_client.QdrantClient("localhost", port=6333)

    # Set up the pipeline
    pipeline = create_query_pipeline(
        qdrant_client=qdrant_client,
        collection_name="my_collection",
        embedding_dimension=768,  # Adjust based on your embedding model
        top_k=10,
        rerank_top_n=5,
        metadata_filter={"category": "science"}  # Example metadata filter
    )

    # Query the pipeline
    query = "What are the latest advancements in quantum computing?"
    result = query_pipeline(pipeline, query)
    print(result)