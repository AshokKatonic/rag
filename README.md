# RAG System - Retrieval-Augmented Generation with Web Scraping

A comprehensive RAG (Retrieval-Augmented Generation) system that scrapes web content, processes it into embeddings, and provides intelligent question-answering capabilities through a REST API.

## 🚀 Features

- **Web Scraping**: Automated scraping of web pages with authentication support
- **Document Processing**: Text chunking and embedding generation using OpenAI
- **Vector Database**: Azure Cosmos DB integration for storing embeddings and metadata
- **RAG Engine**: LangChain-powered question-answering system
- **REST API**: FastAPI-based API with streaming and non-streaming responses
- **Metadata Management**: Comprehensive tracking of document sources and chunks

## 📁 Project Structure

```
Rag/
├── src/
│   ├── settings.py              # Configuration settings
│   ├── web_scraper.py           # Web scraping functionality
│   ├── text_chunker.py          # Text splitting utilities
│   ├── embedding_service.py     # OpenAI embeddings service
│   ├── vector_database.py       # Cosmos DB vector operations
│   ├── metadata_manager.py      # Metadata tracking system
│   └── rag_engine.py           # Core RAG functionality
├── api.py                       # FastAPI application
├── chat.py                      # Client for testing API
├── reload_rag.py               # Data reload script
├── scraping_config.json        # Scraping configuration
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**
   ```bash
   playwright install
   ```

4. **Set up environment variables**
   Create a `.env` file with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   AZURE_COSMO_URL=your_cosmos_db_endpoint
   AZURE_COSMO_KEY=your_cosmos_db_key
   ```

## ⚙️ Configuration

### Scraping Configuration (`scraping_config.json`)
- **target_urls**: List of URLs to scrape
- **authentication**: Login credentials and form selectors
- **scraping**: Browser settings and content processing options
- **output**: Format and metadata inclusion settings

### System Settings (`src/settings.py`)
- **CHUNK_SIZE**: Text chunk size (default: 512)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 80)
- **EMBEDDING_MODEL**: OpenAI embedding model
- **LLM_MODEL**: Language model for responses
- **TOP_K_CHUNKS**: Number of relevant chunks to retrieve

## 🚀 Usage

### 1. Initialize the Vector Store

First, scrape and process documents into the vector database:

```bash
python reload_rag.py
```

This will:
- Clear existing data
- Scrape configured websites
- Process documents into chunks
- Generate embeddings
- Store everything in Cosmos DB

### 2. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Test the System

Use the provided client:

```bash
python chat.py
```

## 📡 API Endpoints

### Chat Completions
- **POST** `/v1/chat/completions`
- **Parameters**:
  - `model`: Model name (e.g., "gpt-4-turbo-preview")
  - `messages`: Array of message objects
  - `temperature`: Response randomness (0.0-1.0)
  - `max_tokens`: Maximum response length
  - `stream`: Enable streaming responses

### Response Format
```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4-turbo-preview",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text"
      },
      "finish_reason": "stop"
    }
  ],
  "sources": ["url1", "url2"],
  "source_count": 2
}

```
