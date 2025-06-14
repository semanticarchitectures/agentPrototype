# PDF RAG Demo - Setup and Usage Instructions

## What You've Created

I've created a comprehensive PDF RAG (Retrieval-Augmented Generation) system with the following components:

### Core Files:
1. **`pdf_rag_demo.py`** - Main interactive demo script
2. **`batch_pdf_rag.py`** - Batch processing script for multiple prompts
3. **`requirements.txt`** - Python dependencies
4. **`README.md`** - Comprehensive documentation

### Configuration Files:
- **`.env`** - Your API keys (already configured with your OpenAI key)
- **`.gitignore`** - Prevents committing sensitive files
- **`pdf_prompts.txt`** - Sample prompts for testing

### Directories:
- **`pdfs/`** - Place your PDF files here
- **`vector_db/`** - Will store the vector database (auto-created)

## Key Features

✅ **PDF Processing**: Automatically reads all PDF files from the `pdfs/` directory  
✅ **Vector Database**: Creates FAISS embeddings for fast similarity search  
✅ **Persistent Storage**: Saves vector database to disk for reuse  
✅ **Interactive Mode**: Chat-like interface for asking questions  
✅ **Batch Processing**: Process multiple prompts from a file  
✅ **Source Tracking**: Shows which documents were used for answers  

## Installation

To install the required dependencies, run:

```bash
pip install python-dotenv langchain langchain-openai langchain-community faiss-cpu pypdf openai
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Interactive Mode
```bash
python pdf_rag_demo.py
```

This starts an interactive session where you can:
- Ask questions about your PDF documents
- Type `sources` to see available PDF files
- Type `quit` or `exit` to end the session

### 2. Batch Processing
```bash
python batch_pdf_rag.py
```

This processes all prompts from `pdf_prompts.txt` and saves results to a JSON file.

## How It Works

1. **Document Loading**: Uses PyPDFLoader to extract text from PDF files
2. **Text Chunking**: Splits documents into 1000-character chunks with 200-character overlap
3. **Embedding**: Creates vector embeddings using OpenAI's embedding model
4. **Vector Storage**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds the 5 most relevant document chunks for each query
6. **Generation**: Uses ChatGPT to generate answers based on retrieved context

## Sample Workflow

1. **Add PDFs**: Place your PDF files in the `pdfs/` directory
2. **First Run**: The system will process all PDFs and create a vector database
3. **Subsequent Runs**: The system loads the existing vector database (much faster)
4. **Query**: Ask questions about your documents in natural language

## Example Queries

- "What are the main topics covered in the documents?"
- "Can you summarize the key findings from the research papers?"
- "What methodologies are discussed in the documents?"
- "Are there any specific recommendations mentioned?"
- "What are the limitations or challenges identified?"

## Performance Notes

- **First run**: Slow (creates embeddings for all PDFs)
- **Subsequent runs**: Fast (loads existing vector database)
- **Large PDFs**: May require more processing time and memory

## Troubleshooting

- **No PDFs found**: Make sure PDF files are in the `pdfs/` directory
- **API key error**: Check that your OpenAI API key is set in `.env`
- **Import errors**: Install missing dependencies with pip
- **Memory issues**: For large PDFs, consider processing fewer files at once

## Next Steps

To test the system:

1. Add some PDF files to the `pdfs/` directory
2. Run `python pdf_rag_demo.py` for interactive mode
3. Or run `python batch_pdf_rag.py` for batch processing

The system is ready to use once you have the dependencies installed and PDF files in place!
