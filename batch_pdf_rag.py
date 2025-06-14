import os
import json
import logging
from datetime import datetime
from pdf_rag_demo import PDFRAGDemo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_prompts_from_file(demo, prompts_file="pdf_prompts.txt", output_file=None):
    """Process prompts from a file and log results."""
    
    if not os.path.exists(prompts_file):
        logger.error(f"Prompts file {prompts_file} not found")
        return
    
    # Read prompts from file
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        logger.error("No prompts found in file")
        return
    
    logger.info(f"Processing {len(prompts)} prompts from {prompts_file}")
    
    # Prepare output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"rag_results_{timestamp}.json"
    
    results = []
    
    # Process each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Processing Prompt {i}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"Question: {prompt}")
        print("-" * 60)
        
        # Get response from RAG system
        result = demo.query(prompt)
        
        if result:
            print(f"Answer: {result['answer']}")
            
            # Add timestamp and prompt number
            result['prompt_number'] = i
            result['timestamp'] = datetime.now().isoformat()
            result['source_count'] = len(result.get('source_documents', []))
            
            results.append(result)
            
            # Show source information
            if result['source_documents']:
                print(f"\nBased on {len(result['source_documents'])} document chunks")
        else:
            print("Error: Failed to get response")
            results.append({
                'prompt_number': i,
                'question': prompt,
                'answer': 'ERROR: Failed to get response',
                'timestamp': datetime.now().isoformat(),
                'source_count': 0
            })
    
    # Save results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total prompts processed: {len(prompts)}")
        print(f"- Successful responses: {len([r for r in results if 'ERROR' not in r.get('answer', '')])}")
        print(f"- Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    print("PDF RAG Batch Processing Demo")
    print("=" * 40)
    
    # Initialize the PDF RAG demo
    demo = PDFRAGDemo()
    
    # Try to load existing vector database first
    if not demo.load_existing_vector_db():
        print("No existing vector database found. Creating new one...")

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
    else:
        print("Loaded existing vector database")
    
    # Create RAG chain
    rag_chain = demo.create_rag_chain()
    if not rag_chain:
        print("Failed to create RAG chain")
        return
    
    # Process prompts from file
    process_prompts_from_file(demo)

if __name__ == "__main__":
    main()
