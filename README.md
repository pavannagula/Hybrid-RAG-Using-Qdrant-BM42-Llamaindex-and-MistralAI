# Hybrid-RAG-Using-Qdrant-BM42-Llamaindex-and-MistralAI
This project provides an overview of a Retrieval-Augmented Generation (RAG) chat application using Qdrant hybrid search, Llamaindex, MistralAI, and re-ranking model. 

Hybrid RAG model combines the strengths of dense vector search and sparse vector search to retrieve relevant documents for a given query. This model uses Qdrant's BM42 approach for sparse vector search, which allows for exact keyword matching and handling of domain-specific terminology. The model also uses dense vector search with the sentence-transformer all-miniLM model to capture semantic relationships and contextual understanding.
The retrieved documents are then re-ranked using a CrossEncoder-based Re-Ranking model to improve the accuracy of the retrieved documents. Finally, the top two documents are used as context to generate a response using MistralAI's 8x7B large language model. The response is then summarized using a response synthesizer to ensure that it is concise and informative.
Overall, the Hybrid RAG model is a powerful solution that addresses the limitations of traditional Semantic Search RAG systems and provides accurate and efficient information retrieval for a wide range of applications.

![Hybrid RAG Architecture](https://github.com/user-attachments/assets/139be431-0019-4246-8eb5-9225191e86fb)


## **Architecture Overview**

1. **Document Pre-Processing**
    - Custom Transformation
    - Sentence Splitter
2. **Indexing**
    - Dense Embedding (for semantic search)
    - Sparse Embedding (for keyword search)
3. **Hybrid Retrieval**
    - Semantic Search (sentence-transformer)
    - Keyword Search (BM42 approach)
    - Reciprocal Rank Fusion
4. **Re-Ranking**
    - Cross Encoder Re-Ranking Model
5. **Response Generation**
    - Prompt Template
    - Llamaindex CustomQueryEngine
    - MistralAI 8x7B Model
6. **Streamlit**

For even more detailed explanation check out this article: https://medium.com/@npavankumar36/hybrid-rag-using-qdrant-bm42-llamaindex-and-mistralai-5d0d51093e8f

Results
![image](https://github.com/user-attachments/assets/8ec32278-5941-45b4-90c4-745c6a458807)

References:
1. https://qdrant.tech/documentation/
2. https://docs.llamaindex.ai/en/stable/
