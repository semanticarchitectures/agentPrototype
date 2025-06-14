# PDF & Web RAG Demo

A Retrieval-Augmented Generation (RAG) system that reads PDF files and web content, creates a vector database, and allows querying the content using natural language.

## Features

- **PDF Processing**: Automatically loads and processes all PDF files from a directory
- **Web Scraping**: Scrapes content from URLs listed in a configuration file
- **Vector Database**: Creates FAISS vector embeddings for efficient similarity search
- **Persistent Storage**: Saves vector database to disk for reuse
- **Interactive Querying**: Chat-like interface for asking questions
- **Batch Processing**: Process multiple prompts from a file
- **Source Tracking**: Shows which documents (PDFs or web pages) were used to answer questions
- **Smart Updates**: Automatically detects when sources have changed

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your-openai-key-here
     ```

3. **Add Content Sources**:
   - Place your PDF files in the `pdfs/` directory
   - Add URLs to scrape in the `urls.txt` file (one URL per line)
   - The system will automatically process all sources

## Usage

### Interactive Mode

Run the main demo for an interactive chat session:

```bash
python pdf_rag_demo.py
```

Commands in interactive mode:
- Type your questions naturally
- Type `sources` to see available PDF files
- Type `quit` or `exit` to end the session

### Batch Processing

Process multiple prompts from a file:

```bash
python batch_pdf_rag.py
```

- Edit `pdf_prompts.txt` to add your questions (one per line)
- Results will be saved to a timestamped JSON file

## How It Works

1. **Document Loading**:
   - Uses PyPDFLoader to extract text from PDF files
   - Uses WebBaseLoader to scrape content from web pages
2. **Text Chunking**: Splits documents into manageable chunks with overlap
3. **Embedding**: Creates vector embeddings using OpenAI's embedding model
4. **Vector Storage**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds relevant document chunks for each query
6. **Generation**: Uses ChatGPT to generate answers based on retrieved context

## File Structure

```
├── pdf_rag_demo.py          # Main interactive demo
├── batch_pdf_rag.py         # Batch processing script
├── pdfs/                    # Directory for PDF files
├── urls.txt                 # URLs to scrape (one per line)
├── vector_db/               # Vector database storage (auto-created)
├── pdf_prompts.txt          # Sample prompts for batch processing
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── README.md               # This file
```

## Example Queries

- "What are the main topics covered in the documents?"
- "Can you summarize the key findings?"
- "What methodologies are discussed?"
- "Are there any specific recommendations?"
- "What are the limitations mentioned?"

## Configuration

You can modify the following parameters in the code:

- **Chunk size**: Default 1000 characters
- **Chunk overlap**: Default 200 characters  
- **Retrieval count**: Default 5 most similar chunks
- **Model**: Default gpt-3.5-turbo (can change to gpt-4)

## Troubleshooting

- **No PDFs found**: Make sure PDF files are in the `pdfs/` directory
- **API key error**: Check that your OpenAI API key is set in `.env`
- **Memory issues**: For large PDFs, consider reducing chunk size or processing fewer files at once
- **Slow processing**: First run creates embeddings (slow), subsequent runs load from disk (fast)
