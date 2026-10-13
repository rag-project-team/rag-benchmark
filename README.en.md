# Comparative Evaluation of Retrieval-Augmented Generation Approaches in Question Answering Systems

This project compares how different Retrieval-Augmented Generation (RAG) configurations perform on document-based question answering. It also includes a question-answering prototype that makes these comparisons possible. Off-the-shelf language models are used; no model is trained from scratch.

## Why this project?

When a large language model is not given a document, it can produce answers whose source is unclear or that are not supported by any document (hallucination). When documents are long and the same concept is expressed in different ways, keyword search also struggles to find the relevant information.

RAG addresses both problems together:

1. Documents are split into chunks, and each chunk is converted into a vector (embedding).
2. When a question arrives, the chunks closest to it in meaning are found (retrieval).
3. These chunks are given to the language model together with the question, and the model produces an answer grounded in the sources.

Every step of this pipeline offers more than one option: which embedding model, which search method, which vector store, which language model. This project examines how these choices change the results through controlled experiments.

## Questions we are trying to answer

**Main question:** How do different RAG components affect answer quality, retrieval success and processing time on Turkish and English documents across different topics?

We break this question down into the following sub-questions:

1. **Does RAG make a difference?** When the same question is asked with and without chunks retrieved from the document, how do answer accuracy and consistency with the sources change? How much does hallucination decrease?
2. **How important is the embedding model?** How do multilingual-e5 and BGE-M3 compare in how successfully they find the correct document chunks?
3. **Does the retrieval method change the outcome?** Is there a difference in retrieval and answer quality between dense vector retrieval and hybrid retrieval (keyword and vector combined)?
4. **What difference does the vector store make?** When Chroma, Qdrant and FAISS are compared with the same chunks, the same embedding and the same top-k, how do indexing time and query time differ?
5. **How does the choice of language model affect the answer?** With the same retrieved chunks, what quality and speed do different off-the-shelf language models achieve?
6. **Is there a difference between Turkish and English?** On paired Turkish and English questions based on the same content, how large is the performance gap, and what is its source (embedding, language model, translation quality)?
7. **When does the system get it wrong?** Do errors come from the retrieval stage (the wrong chunk was retrieved) or from the generation stage (the right chunk was provided but the model answered incorrectly)?

## Approach

Instead of trying every possible combination, a baseline configuration is kept fixed and components are changed one at a time to measure their effects separately. A few selected combinations are then compared with each other.

To keep topic and content differences from being confused with language effects, the Turkish and English experiments are run on the same document content. Limitations arising from translation are reported.

## How do we evaluate?

- **RAGAS metrics:** Faithfulness (is the answer consistent with the retrieved sources), Answer Relevancy (is the answer relevant to the question), Context Precision and Context Recall (were the right chunks found)
- **Response time:** Retrieval and generation times are measured separately
- **Reference answers and human review:** Automatic evaluation is not treated as the only evidence of correctness. Two people independently review selected answers, and disagreements are recorded.

## System to be developed

The comparisons will be available in a web application:

- The user uploads a document and asks questions about it
- The user chooses among different RAG configurations
- For the same question, the user sees the answers, source chunks and response times of different configurations side by side
- As a researcher, the user starts batch experiments and exports the results as CSV

Technologies: Python, Streamlit (front end), FastAPI (back end), SQLite (experiment records), pandas (result analysis).

## Scope

- No claim is made to have developed a new base model or algorithm. The contribution is the experimental design, comparative findings and error analysis.
- Only publicly available documents that are free to use are used in the experiments.
- OCR and additional file formats are out of scope and will be addressed if time permits.

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. https://arxiv.org/abs/2312.10997
3. Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. https://aclanthology.org/2020.emnlp-main.550/
4. Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. https://arxiv.org/abs/2309.15217
5. Wang et al. (2024). Multilingual E5 Text Embeddings: A Technical Report. https://arxiv.org/abs/2402.05672
6. Chen et al. (2024). BGE M3-Embedding. https://arxiv.org/abs/2402.03216
7. Douze et al. (2024). The Faiss library. https://arxiv.org/abs/2401.08281
