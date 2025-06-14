import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

# Validate API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is required.")
    print("Please set it in your .env file or as an environment variable.")
    exit(1)

def create_rag_database(docs_dir="documents"):
    """Create a RAG database from documents in the specified directory."""
    print(f"Loading documents from {docs_dir}...")
    
    # Load documents
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    print(f"Creating vector store with {len(splits)} document chunks...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    return vector_store

def create_rag_chain(vector_store):
    """Create a RAG chain using the vector store."""
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """)
    
    # Create chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    return rag_chain

def process_prompts(rag_chain, prompts_file="prompts.txt"):
    """Process prompts from a file using the RAG chain."""
    print(f"Reading prompts from {prompts_file}...")
    
    # Read prompts from file
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": prompt})
        
        # Log result
        print(f"Response: {response['answer']}")

def main():
    # Create documents directory if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' directory. Please add your text files there.")
        print("Then create a 'prompts.txt' file with your prompts (one per line).")
        return
    
    # Check if prompts file exists
    if not os.path.exists("prompts.txt"):
        print("Please create a 'prompts.txt' file with your prompts (one per line).")
        return
    
    # Create RAG database
    vector_store = create_rag_database()
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store)
    
    # Process prompts
    process_prompts(rag_chain)

if __name__ == "__main__":
    main()