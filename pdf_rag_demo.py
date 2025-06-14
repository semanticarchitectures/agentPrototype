import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
import time

print("Starting PDF RAG Demo...")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Validate API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is required.")
    print("Please set it in your .env file or as an environment variable.")
    exit(1)

print(os.getenv("OPENAI_API_KEY"))

class PDFRAGDemo:
    def __init__(self, pdf_dir="pdfs", vector_db_path="vector_db", urls_file="urls.txt"):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.urls_file = urls_file
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vector_store = None
        self.rag_chain = None
        
    def load_pdfs(self):
        """Load PDF files from the specified directory."""
        logger.info(f"Loading PDF files from {self.pdf_dir}...")
        
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
            logger.warning(f"Created {self.pdf_dir} directory. Please add PDF files there.")
            return []
        
        # Get all PDF files in the directory
        pdf_files = list(Path(self.pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load documents from PDFs
        documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading {pdf_file.name}...")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_path'] = str(pdf_file)
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def load_urls(self):
        """Load content from URLs listed in the urls file."""
        logger.info(f"Loading URLs from {self.urls_file}...")

        if not os.path.exists(self.urls_file):
            logger.info(f"No URLs file found at {self.urls_file}")
            return []

        # Read URLs from file
        with open(self.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        if not urls:
            logger.info("No URLs found in file")
            return []

        logger.info(f"Found {len(urls)} URLs to process")

        documents = []
        for url in urls:
            try:
                logger.info(f"Loading content from {url}...")
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Add metadata about the source URL
                for doc in docs:
                    doc.metadata['source_type'] = 'web'
                    doc.metadata['source_url'] = url
                    doc.metadata['source_file'] = url.split('/')[-1] or url

                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {url}")

            except Exception as e:
                logger.error(f"Error loading {url}: {e}")
                continue

        logger.info(f"Total web documents loaded: {len(documents)}")
        return documents

    def load_all_documents(self):
        """Load documents from both PDFs and URLs."""
        all_documents = []

        # Load PDFs
        pdf_docs = self.load_pdfs()
        all_documents.extend(pdf_docs)

        # Load URLs
        web_docs = self.load_urls()
        all_documents.extend(web_docs)

        logger.info(f"Total documents from all sources: {len(all_documents)}")
        logger.info(f"  - PDF documents: {len(pdf_docs)}")
        logger.info(f"  - Web documents: {len(web_docs)}")

        return all_documents
    
    def create_vector_database(self, documents, save_to_disk=True):
        """Create a vector database from the loaded documents."""
        if not documents:
            logger.error("No documents to process")
            return None
        
        logger.info("Splitting documents into chunks...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        # Create vector store
        logger.info("Creating vector embeddings...")
        start_time = time.time()
        
        try:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            if save_to_disk:
                logger.info(f"Saving vector database to {self.vector_db_path}")
                self.vector_store.save_local(self.vector_db_path)
            
            end_time = time.time()
            logger.info(f"Vector database created in {end_time - start_time:.2f} seconds")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            return None
    
    def load_existing_vector_db(self):
        """Load an existing vector database from disk."""
        if os.path.exists(self.vector_db_path):
            try:
                logger.info(f"Loading existing vector database from {self.vector_db_path}")
                self.vector_store = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector database loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
                return False
        return False

    def check_sources_changed(self):
        """Check if PDFs or URLs have been added, removed, or modified since last database creation."""
        if not os.path.exists(self.vector_db_path):
            return True  # No database exists

        # Get database creation time
        db_time = os.path.getmtime(self.vector_db_path)

        # Check if any PDF is newer than the database
        if os.path.exists(self.pdf_dir):
            pdf_files = list(Path(self.pdf_dir).glob("*.pdf"))
            for pdf_file in pdf_files:
                if os.path.getmtime(pdf_file) > db_time:
                    logger.info(f"PDF {pdf_file.name} is newer than vector database")
                    return True

        # Check if URLs file is newer than the database
        if os.path.exists(self.urls_file):
            if os.path.getmtime(self.urls_file) > db_time:
                logger.info(f"URLs file {self.urls_file} is newer than vector database")
                return True

        return False
    
    def create_rag_chain(self):
        """Create a RAG chain using the vector store."""
        if not self.vector_store:
            logger.error("Vector store not available")
            return None
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context from PDF documents.
        
        Context from PDFs:
        {context}
        
        Question: {input}
        
        Instructions:
        - Answer the question based on the provided context
        - If the context doesn't contain enough information, say so
        - Include relevant details from the source documents
        - Be concise but comprehensive
        
        Answer:
        """)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            output_parser=StrOutputParser()
        )
        
        # Create retrieval chain
        self.rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        logger.info("RAG chain created successfully")
        return self.rag_chain
    
    def query(self, question):
        """Query the RAG system with a question."""
        if not self.rag_chain:
            logger.error("RAG chain not initialized")
            return None
        
        try:
            logger.info(f"Processing query: {question}")
            response = self.rag_chain.invoke({"input": question})
            
            return {
                "question": question,
                "answer": response["answer"],
                "source_documents": response.get("context", [])
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return None
    
    def interactive_query(self):
        """Start an interactive query session."""
        print("\n" + "="*50)
        print("PDF RAG Demo - Interactive Query Session")
        print("="*50)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'sources' to see available PDF sources")
        print("-"*50)
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if question.lower() == 'sources':
                self.show_sources()
                continue
            
            if not question:
                continue
            
            result = self.query(question)
            if result:
                print(f"\nQuestion: {result['question']}")
                print(f"Answer: {result['answer']}")
                
                # Show source information
                if result['source_documents']:
                    print(f"\nBased on {len(result['source_documents'])} document chunks")
    
    def show_sources(self):
        """Show available sources in the vector database."""
        if not self.vector_store:
            print("No vector database available")
            return

        try:
            print("\nAvailable sources:")

            # Show PDF sources
            if os.path.exists(self.pdf_dir):
                pdf_files = list(Path(self.pdf_dir).glob("*.pdf"))
                if pdf_files:
                    print("\nPDF Documents:")
                    for i, pdf_file in enumerate(pdf_files, 1):
                        print(f"  {i}. {pdf_file.name}")

            # Show URL sources
            if os.path.exists(self.urls_file):
                with open(self.urls_file, 'r') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                if urls:
                    print("\nWeb Sources:")
                    for i, url in enumerate(urls, 1):
                        print(f"  {i}. {url}")

            if not pdf_files and not urls:
                print("No sources found")

        except Exception as e:
            print(f"Error retrieving sources: {e}")

def main():
    # Check for command line flags
    force_rebuild = "--rebuild" in sys.argv or "--force" in sys.argv
    auto_update = "--auto-update" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("PDF RAG Demo Usage:")
        print("  python3 pdf_rag_demo.py                 # Normal mode")
        print("  python3 pdf_rag_demo.py --rebuild       # Force rebuild database")
        print("  python3 pdf_rag_demo.py --auto-update   # Auto-update if PDFs changed")
        print("  python3 pdf_rag_demo.py --help          # Show this help")
        return

    # Initialize the PDF RAG demo
    demo = PDFRAGDemo()

    # Determine if we should rebuild the database
    should_rebuild = False

    if force_rebuild:
        print("Force rebuilding vector database...")
        should_rebuild = True
        # Remove existing database
        import shutil
        if os.path.exists(demo.vector_db_path):
            shutil.rmtree(demo.vector_db_path)
    elif auto_update and demo.check_sources_changed():
        print("Sources have changed. Updating vector database...")
        should_rebuild = True
        import shutil
        if os.path.exists(demo.vector_db_path):
            shutil.rmtree(demo.vector_db_path)
    elif not demo.load_existing_vector_db():
        print("No existing vector database found. Creating new one...")
        should_rebuild = True
    else:
        print("Loaded existing vector database")

    if should_rebuild:
        # Load all documents (PDFs and URLs) and create vector database
        documents = demo.load_all_documents()

        if not documents:
            print("No documents found. Please add PDF files to the 'pdfs' directory or URLs to 'urls.txt'.")
            return

        # Create vector database
        vector_store = demo.create_vector_database(documents)
        if not vector_store:
            print("Failed to create vector database")
            return

    # Create RAG chain
    rag_chain = demo.create_rag_chain()
    if not rag_chain:
        print("Failed to create RAG chain")
        return

    # Start interactive session
    demo.interactive_query()

if __name__ == "__main__":
    main()
