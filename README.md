
RAG Research Assistant - LangChain + FAISS + GPT-3.5

Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to query PDF documents using natural language. The system retrieves the most relevant text chunks from the PDFs and generates grounded answers using OpenAI's GPT-3.5 model. The answers include references to the source text to ensure transparency and reduce hallucinations.

Features
1. PDF loading and text extraction using PyMuPDF.
2. Text chunking with overlap to preserve context.
3. Embedding generation using OpenAI's text-embedding-3-small model.
4. Vector storage and retrieval using FAISS for fast similarity search.
5. Question answering powered by OpenAI's GPT-3.5 with relevant sources displayed.
6. Interactive chat loop for continuous Q&A with the PDF content.

Folder Structure
project_root/
    RAG_assistant_GIT.ipynb     # Jupyter Notebook version
    RAG_assistant_GIT.py        # Python script version
    
Installation
1. Install Python 3.9 or later.
2. Install the required dependencies:
   pip install langchain openai faiss-cpu tiktoken pymupdf langchain-community

3. Obtain an OpenAI API key from https://platform.openai.com/ and set it in the script or as an environment variable.

Usage
1. Place your PDF files in the docs/ folder.
2. Run the Jupyter Notebook or the Python script to:
   - Load and preprocess PDFs
   - Generate embeddings and store them in FAISS
   - Ask questions and receive answers with sources

Example
Question: What is the PALMS process about?
Answer: The PALMS process is about adapting language models to society by fine-tuning them on a dataset that reflects predetermined values, aiming to reduce harmful and biased outputs.

Configuration
- chunk_size: Adjust to control the size of text chunks for embedding.
- chunk_overlap: Adjust to retain context between chunks.
- k: Number of top similar chunks retrieved for answering each question.

Notes
- Keep your real API key private. Do not commit it to public repositories.
- Ensure that the docs/ folder contains the PDFs you want to query before running the script.

License
This project is provided for educational and demonstration purposes. Use at your own risk.
