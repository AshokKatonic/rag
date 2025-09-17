import asyncio
import os
import json
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text
import nest_asyncio
from datetime import datetime
from langchain.schema import Document
from . import text_chunker, embedding_service, vector_database
from .metadata_manager import generate_chunk_id, add_chunk
from dotenv import load_dotenv

nest_asyncio.apply()

DEFAULT_CONFIG = {
    "target_urls": [
        "https://partners.katonic.ai/dashboard/",
        "https://partners.katonic.ai/infrastructure/?type=on-prem-normal",
        "https://partners.katonic.ai/infrastructure/?type=on-prem-multitenancy",
        "https://partners.katonic.ai/infrastructure/?type=airgap",
        "https://partners.katonic.ai/infrastructure/?type=airgap-multitenancy",
        "https://partners.katonic.ai/installation/?type=airgap-multitenancy"
    ],
    "authentication": {
        "enabled": True,
        "login_url": "https://partners.katonic.ai/dashboard/",
        "email_field": 'input[name="email"]',
        "password_field": 'input[name="password"]',
        "submit_button": 'button[type="submit"]',
        "email": "admin@katonic.ai",
        "password": "admin123"
    },
    "scraping": {
        "wait_for": "networkidle",
        "timeout": 30000,
        "remove_scripts": True,
        "remove_styles": True,
        "follow_links": False,
        "max_depth": 1
    },
    "output": {
        "format": "markdown",
        "include_metadata": True,
        "clean_text": True
    }
}

async def scrape_page_direct(page, url: str, config):
    try:
        scraping_config = config.get("scraping", {})
        wait_for = scraping_config.get("wait_for", "networkidle")
        timeout = scraping_config.get("timeout", 30000)
        
        await page.goto(url, wait_until=wait_for, timeout=timeout)
        content = await page.content()
        
        soup = BeautifulSoup(content, "html.parser")
        
        if scraping_config.get("remove_scripts", True):
            for script in soup(["script"]):
                script.decompose()
        
        if scraping_config.get("remove_styles", True):
            for style in soup(["style"]):
                style.decompose()
        
        title = soup.find("title")
        title_text = title.get_text().strip() if title else "Untitled"
        
        if config.get("output", {}).get("format") == "html":
            page_content = str(soup)
        else:
            page_content = html2text.html2text(str(soup))
        
        return {
            "url": url,
            "title": title_text,
            "content": page_content,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
    except Exception as e:
        return {
            "url": url,
            "title": "Error",
            "content": f"Failed to scrape {url}: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }

async def authenticate_direct(page, config):
    auth_config = config.get("authentication", {})
    
    if not auth_config.get("enabled", False):
        return True
    
    try:
        login_url = auth_config.get("login_url")
        email_field = auth_config.get("email_field")
        password_field = auth_config.get("password_field")
        submit_button = auth_config.get("submit_button")
        email = auth_config.get("email")
        password = auth_config.get("password")
        
        print(f"Authenticating at {login_url}...")
        await page.goto(login_url)
        
        if email_field and email:
            await page.fill(email_field, email)
        if password_field and password:
            await page.fill(password_field, password)
        if submit_button:
            await page.click(submit_button)
            await page.wait_for_load_state("networkidle")
        
        print("Authentication completed")
        return True
        
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

async def scrape_to_documents(config = None):
    if config is None:
        config = DEFAULT_CONFIG
    
    target_urls = config.get("target_urls", [])
    if not target_urls:
        print("No target URLs configured!")
        return []
    
    print(f"Starting to scrape {len(target_urls)} URLs directly to documents...")
    
    documents = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        auth_success = await authenticate_direct(page, config)
        if not auth_success:
            print("Authentication failed, continuing without auth...")
        
        for i, url in enumerate(target_urls, 1):
            print(f"\n[{i}/{len(target_urls)}] Scraping {url}")
            scraped_data = await scrape_page_direct(page, url, config)
            
            if scraped_data["success"]:
                content = scraped_data["content"]
                if config.get("output", {}).get("include_metadata", True):
                    content = f"Source: {url}\nTitle: {scraped_data['title']}\n\n{content}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': url,
                        'title': scraped_data['title'],
                        'timestamp': scraped_data['timestamp'],
                        'scraped_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                print(f"✓ Created document for {url}")
            else:
                print(f"✗ Failed to scrape {url}: {scraped_data.get('error', 'Unknown error')}")
        
        await browser.close()
    
    print(f"\n{'='*50}")
    print(f"Scraping completed: {len(documents)} documents created")
    return documents

async def scrape_and_process_to_rag(config = None, api_key: str = None):

    if not api_key:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
    
    documents = await scrape_to_documents(config)
    
    if not documents:
        print("No documents scraped!")
        return None
    
    embeddings_model = embedding_service.create_embeddings_model(api_key)
    
    text_splitter = text_chunker.create_text_splitter()
    chunk_documents = []
    
    for doc in documents:
        doc_chunks = text_chunker.split_into_chunks(doc.page_content, text_splitter)
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
    
    print(f"Creating vector store with {len(chunk_documents)} document chunks...")
    
    vector_database.create_vector_store(chunk_documents, embeddings_model)
    print("Vector store created successfully!")
    print(f"Created {len(chunk_documents)} chunk metadata records")
    
    return True

async def main():
    documents = await scrape_to_documents()
    print(f"\nScraped {len(documents)} documents:")
    for doc in documents:
        print(f"- {doc.metadata['source']}: {doc.metadata['title']}")

if __name__ == "__main__":
    asyncio.run(main())
