from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from rerank import reranking
from Retriever import Hybrid_search
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class QueryPipelineBuilder:
    def __init__(
        self,
        search: Hybrid_search,
        reranker: reranking,
        file_names: str,
        
    ):
        self.search = search
        self.reranker = reranker
        self.file_names = file_names
        

        self.prompt_str = """You are an AI assistant specializing in explaining complex topics related to adaptive retrieval
        and AI-powered search systems. Your task is to provide a clear, concise, and informative explanation based on the
        following context and query.

        Context:
        {context_str}

        Query: {query_str}

        Please follow these guidelines in your response:
        1. Start with a brief overview of the concept mentioned in the query.
        2. Explain how this concept relates to adaptive retrieval or AI-powered search systems.
        3. Highlight any key advantages or innovations this concept brings to information retrieval.
        4. If relevant, mention how this concept compares to traditional retrieval methods.
        5. Provide at least one concrete example or use case to illustrate the concept.
        6. If there are any limitations or challenges associated with this concept, briefly mention them.
        7. Conclude with a sentence about the potential future impact or applications of this concept.

        Your explanation should be informative yet accessible, suitable for someone with a basic understanding of AI and information
        retrieval. If the query asks for information not present in the context, please state that you don't have enough information
        to provide a complete answer, and only respond based on the given context.

        Response:
        """

    def query_pipeline(self, query: str) -> QueryPipeline:
        # Set up LLM
        llm = Groq(
            model="mixtral-8x7b-32768",
            api_key=os.getenv('groq_api_key')
        )

        # Set up prompt template
        prompt_tmpl = PromptTemplate(self.prompt_str)

        # Set up summarizer
        summarizer = TreeSummarize(llm=llm)

        # Call the retriever and reranker functions explicitly
        metadata_filter = self.search.metadata_filter(self.file_names)
        results = self.search.query_hybrid_search(query, metadata_filter)
        reranked_documents = self.reranker.rerank_documents(query, results)

        # Include the reranked documents in the context
        context = "\n\n".join(reranked_documents)
        prompt = prompt_tmpl.partial_format(context_str=context, query_str=query)

        # Create query pipeline
        pipeline = QueryPipeline()

        # Add modules to the pipeline
        pipeline.add_modules({
            "prompt": {"module": prompt, "input_key": "query_str"},
            "llm": llm,
            "summarizer": summarizer
        })

        # Define links between modules
        pipeline.add_link("prompt", "llm")
        pipeline.add_link("llm", "summarizer")

        return pipeline




if __name__ == '__main__':
    # Create a Hybrid_search instance
    search = Hybrid_search()

    # Create a reranking instance
    reranker = reranking()

    # Create a query pipeline builder
    pipeline_builder = QueryPipelineBuilder(search, reranker, file_names="Adaptive-RAG.pdf")

    # Create a query pipeline
    pipeline = pipeline_builder.query_pipeline(query="Explain adaptive retrieval and its advantages.")

    # Run a query through the pipeline
    response = pipeline.run()

    # Print the response
    print(response)

