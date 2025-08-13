#!/usr/bin/env python
# coding: utf-8

# # Step 1 — Install Dependencies
# 

# In[1]:


get_ipython().system('pip install langchain openai faiss-cpu tiktoken pymupdf')



# In[10]:


get_ipython().system('pip install -U langchain-community')


# # Step 2 — Set API Key and Model

# In[2]:


import openai

# Paste your OpenAI API key 
openai.api_key = "YOUR_API_KEY_HERE"

# Model
MODEL_NAME = "gpt-3.5-turbo"


# # Step 3 — Load PDFs from Folder

# In[7]:


import fitz  # PyMuPDF
import os

def load_pdfs_from_folder(folder_path):
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            pdf_doc = fitz.open(file_path)
            text = ""
            for page in pdf_doc:
                text += page.get_text()
            all_texts.append({"filename": filename, "text": text})
    return all_texts

# Load all PDFs from docs folder
pdf_texts = load_pdfs_from_folder("docs")

# Show first 500 characters of the first PDF
if pdf_texts:
    print(f"Loaded {len(pdf_texts)} PDFs.")
    print(f"Example from: {pdf_texts[0]['filename']}\n")
    print(pdf_texts[0]['text'][:500])
else:
    print("No PDFs found in docs/ folder.")


# # Step 4 — Chunk Text

# In[8]:


from langchain.text_splitter import CharacterTextSplitter

# Combine all PDFs into one big text list
all_text = ""
for pdf in pdf_texts:
    all_text += pdf["text"] + "\n"

# Create the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,     # each chunk ~1000 characters
    chunk_overlap=150,   # overlap between chunks
    length_function=len
)

# Split into chunks
chunks = text_splitter.split_text(all_text)

print(f"Total chunks created: {len(chunks)}")
print("\nFirst chunk preview:\n")
print(chunks[0])


# # Step 5 — Create Embeddings and Store in FAISS

# In[11]:


from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Create embeddings object
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai.api_key)

# Create FAISS vector store from chunks
vectorstore = FAISS.from_texts(chunks, embeddings)

# Save the FAISS index locally
vectorstore.save_local("faiss_index")

print(" Embeddings created and FAISS index built!")


# # Step 6 — Retrieval + GPT Answering

# In[12]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Create retriever from FAISS index
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create the GPT model
llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=openai.api_key)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "What is the PALMS process about?"
result = qa_chain({"query": query})

# Print the answer
print("Answer:\n", result["result"])

# Show sources
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:200]}...")  # show first 200 chars of each source


# # Step 7 — Interactive Chat with PDF
# 

# In[ ]:


while True:
    user_q = input("\nAsk a question (or type 'exit' to quit): ")
    if user_q.lower() in ["exit", "quit"]:
        break
    
    result = qa_chain({"query": user_q})
    print("\nAnswer:\n", result["result"])
    
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"Source {i}: {doc.page_content[:200]}...")


# In[ ]:




