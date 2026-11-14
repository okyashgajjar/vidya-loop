# Introduction to RAG

## What is RAG?
Retrieval-Augmented Generation (RAG) is a method that retrieves useful outside information and gives it to a model before it writes an answer. Retrieval means finding useful stored information. Think of it as an open-book exam for AI. A regular AI answers questions from memory alone. RAG lets the AI look up facts in a textbook first.

## Why does RAG matter?
AI models sometimes make mistakes. When an AI invents facts, we call it a hallucination, meaning a confident but false answer. RAG stops hallucinations by providing fresh facts from verified notes. This keeps answers accurate and trustworthy.

## How does RAG work?
1. **Store documents**: We split long documents into chunks. Chunking means breaking a big document into smaller pieces. Each chunk is stored in a vector database, which is a database that groups text by meaning.
2. **Find relevant chunks**: The system performs retrieval, meaning it finds the best matching paragraph for the question. It uses embeddings, which are lists of numbers that capture meaning.
3. **Add facts to the prompt**: The prompt augmentation step (meaning adding retrieved facts to your question) prepares the complete prompt.
4. **Generate the answer**: The AI reads both the question and the facts to give an accurate answer.

## Practical example: RAG
Imagine a student preparing for board exams. An open-book test is always safer than guessing from memory. The student looks up the answer in a textbook before writing it. RAG does the same for AI: it checks a knowledge base before answering, so the answer is grounded in real facts rather than the model's memory alone.

**Glossary:**
- **Retrieval-augmented generation (RAG):** A method that retrieves useful outside information and gives it to a model before it writes an answer.
- **Vector database:** A database that stores information by meaning instead of exact keywords.
- **Embedding:** A list of numbers that captures the meaning of text.
- **Chunking:** Breaking down big documents into smaller, easy-to-read pieces.
- **Hallucination:** When an AI makes up confident-sounding answers that are actually incorrect.
