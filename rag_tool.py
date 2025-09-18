from mcp.server.fastmcp import FastMCP
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import glob 
from dotenv import load_dotenv
load_dotenv()
mcp=FastMCP('RAG')


vectorstore=None


def build_vectorStore():
    global vectorstore
    docs_dir="docs"
    file_paths=glob.glob(os.path.join(docs_dir, "*.txt"))
    if not file_paths:
        raise ValueError(f"No .txt files found in directory: {docs_dir}")
    
    documents=[]
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content=file.read()
            documents.append({"text":content, 'source':os.path.basename(path)})
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts=[]
    metadatas=[]
    for doc in documents:
        splits=splitter.split_text(doc['text'])
        texts.extend(splits)
        metadatas.extend([{'source': doc['source']}] * len(splits))
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, emb, metadatas=metadatas)


@mcp.tool()
def rag_tool(query: str):
    """Performs semantic search over the vectorstore built from docs/*.txt files"""
    global vectorstore
    if vectorstore is None:
        build_vectorStore()

    docs=vectorstore.similarity_search(query, k=3)
    return "\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs])

if __name__ == "__main__":
    mcp.run(transport=("stdio"))