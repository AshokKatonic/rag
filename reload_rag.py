import asyncio
import os
from dotenv import load_dotenv
from src.web_scraper import scrape_to_documents
from src.vector_database import clear_vector_store, create_vector_store
from src.embedding_service import create_embeddings_model
from src.text_chunker import create_text_splitter, split_into_chunks
from src.metadata_manager import clear_all_metadata, generate_chunk_id, add_chunk
from langchain.schema import Document
from src.vector_database import get_document_count

async def reload_vector_store():
    print("Starting Vector Store Reload Process")
    print("=" * 50)
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    print("\n1. Clearing existing vector store and metadata...")
    try:
        cleared_count = clear_vector_store()
        clear_all_metadata()
        print(f"Cleared {cleared_count} documents from Cosmos DB")
        print("Cleared all metadata")
    except Exception as e:
        print(f"Error clearing vector store: {e}")
        return False
    
    print("\n2. Scraping fresh documents from websites...")
    try:
        documents = await scrape_to_documents()
        if not documents:
            print("No documents were scraped!")
            return False
        
        print(f"Successfully scraped {len(documents)} documents")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc.metadata['source']}")
            print(f"      Title: {doc.metadata.get('title', 'N/A')}")
            print(f"      Content length: {len(doc.page_content)} characters")
    except Exception as e:
        print(f"Error scraping documents: {e}")
        return False
    
    print("\n3. Processing documents into chunks...")
    try:
        embeddings_model = create_embeddings_model(api_key)
        text_splitter = create_text_splitter()
        chunk_documents = []
        
        for doc in documents:
            doc_chunks = split_into_chunks(doc.page_content, text_splitter)
            source_url = doc.metadata.get('source', 'unknown')
            total_chunks = len(doc_chunks)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'source': source_url,
                        'chunk_index': i,
                        'title': doc.metadata.get('title', ''),
                        'timestamp': doc.metadata.get('timestamp', ''),
                        'scraped_at': doc.metadata.get('scraped_at', '')
                    }
                )
                chunk_documents.append(chunk_doc)
                
                chunk_id = generate_chunk_id(source_url, i)
                add_chunk(chunk_id, source_url, total_chunks)
        
        print(f" Created {len(chunk_documents)} document chunks")
    except Exception as e:
        print(f" Error processing documents: {e}")
        return False
    
    print("\n4. Generating embeddings and storing in Cosmos DB...")
    try:
        create_vector_store(chunk_documents, embeddings_model)
        print(" Successfully created vector store with embeddings")
        print(f" Stored {len(chunk_documents)} document chunks in Cosmos DB")
    except Exception as e:
        print(f" Error creating vector store: {e}")
        return False
    
    print("\n5. Verifying the reload...")
    try:
        doc_count = get_document_count()
        print(f" Verification complete: {doc_count} documents in vector store")
    except Exception as e:
        print(f"⚠ Warning: Could not verify document count: {e}")
    
    print("\n Vector store reload completed successfully!")
    print("=" * 50)
    print("The RAG system is now ready with fresh data.")
    print("You can now use the API endpoints to ask questions.")
    
    return True

async def main():
    try:
        success = await reload_vector_store()
        if success:
            print("\n Reload process completed successfully!")
        else:
            print("\n Reload process failed!")
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠Reload process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n Unexpected error during reload: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())


