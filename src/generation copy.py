from llama_index.core import QueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import BaseNodePostprocessor
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from rerank import reranking
from Retriever import Hybrid_search
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os


class CustomRetriever(BaseRetriever):
    def __init__(self, search, metadata_filter):
        self.search = search
        self.metadata_filter = metadata_filter

    def _retrieve(self, query, **kwargs):
        results = self.search.query_hybrid_search(query, self.metadata_filter)
        return results

class CustomReranker(BaseNodePostprocessor):
    def __init__(self, reranker):
        self.reranker = reranker

    def postprocess_nodes(self, nodes, query, **kwargs):
        reranked_documents = self.reranker.rerank_documents(query, nodes)
        return reranked_documents

class CustomQueryEngine(QueryEngine):
    def __init__(self, llm, prompt_tmpl, search, reranker, filename):
        self.llm = llm
        self.prompt_tmpl = prompt_tmpl
        self.search = search
        self.reranker = reranker
        self.filename = filename

    def query(self, query_str):
        # Create retriever
        metadata_filter = self.search.metadata_filter(self.filename)
        retriever = CustomRetriever(self.search, metadata_filter)

        # Retrieve initial results
        initial_results = retriever.retrieve(query_str)

        # Rerank results
        reranker = CustomReranker(self.reranker)
        reranked_results = reranker.postprocess_nodes(initial_results, query_str)

        # Prepare context
        context = "".join(reranked_results)

        # Generate prompt
        prompt = self.prompt_tmpl.format(context_str=context, query_str=query_str)

        # Synthesize response
        summarizer = TreeSummarize(llm=self.llm)
        response = summarizer.synthesize(prompt, query_str)

        return response

prompt_str = """You are an AI assistant specializing in explaining complex topics related to AI-powered RAG systems. Your task is to provide a clear, concise, and informative explanation based on the following context and query.

Context:
{context_str}

Query: {query_str}

Please follow these guidelines in your response:
1. Start with a brief overview of the concept mentioned in the query.
2. Provide at least one concrete example or use case to illustrate the concept.
3. If there are any limitations or challenges associated with this concept, briefly mention them.
4. Conclude with a sentence about the potential future impact or applications of this concept.

Your explanation should be informative yet accessible, suitable for someone with a basic understanding of AI and RAG. If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer, and only respond based on the given context.

Response:
"""

# Usage
query_engine = CustomQueryEngine(
    llm= Groq(model="mixtral-8x7b-32768", api_key=os.getenv('groq_api_key')),
    prompt_tmpl = PromptTemplate(prompt_str),
    search=Hybrid_search,
    reranker=reranking,
    filename="Adaptive-RAG.pdf"
)

if __name__ == '__main__':
    query="Explain adaptive retrieval and its advantages."
    response = query_engine.query(query)