# RAG-Using-Hybrid-Search-and-Re-Ranking
This project provides an overview and implementation steps for constructing a Retrieval-Augmented Generation (RAG) application using a hybrid search and re-ranking retriever. The architecture leverages both semantic and keyword search techniques to enhance query processing and retrieval accuracy. Additionally, the re-ranking step ensures better retrieval by extracting the relevant documents, leading to more accurate and contextually relevant responses.

## **Architecture Overview**

1. **Document Ingestion**
    - Document Loader
    - Text Chunking
    - Embedding Generation
2. **Indexing**
    - Vector Store (for semantic search)
    - Inverted Index (for keyword search)
3. **Query Processing**
    - Query Understanding
    - Query Expansion (optional)
4. **Hybrid Retrieval**
    - Semantic Search
    - Keyword Search (e.g., BM25)
    - Ensemble Retriever
5. **Re-Ranking**
    - Re-Ranking Model
6. **Context Formation**
7. **LLM Integration**
8. **Response Generation**
