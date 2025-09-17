import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from . import text_chunker
from . import embedding_service
from . import vector_database
from . import settings
from .metadata_manager import generate_chunk_id, add_chunk, clear_all_metadata

def setup_rag_system(force_reload=False, api_key=None):
    if not api_key:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")

    embeddings_model = embedding_service.create_embeddings_model(api_key)

    vector_store_exists = vector_database.load_vector_store(embeddings_model)
    
    if not vector_store_exists:
        print("No vector store found. Please run refresh_data.py to create one.")
        return None

    llm = ChatOpenAI(
        model_name=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        openai_api_key=api_key
    )

    prompt = PromptTemplate(
        template=settings.RAG_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_database.create_retriever(embeddings_model, search_kwargs={"k": settings.TOP_K_CHUNKS}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# def setup_rag_system_from_documents(documents: list, force_reload=False, api_key=None):
#     if not api_key:
#         load_dotenv()
#         api_key = os.getenv('OPENAI_API_KEY')
#         if not api_key:
#             raise ValueError("OpenAI API key not found")

#     embeddings_model = embedding_service.create_embeddings_model(api_key)

#     if force_reload:
#         vector_database.clear_vector_store()
#         clear_all_metadata()
#         print("Cleared existing vector store and metadata")

#     text_splitter = text_chunker.create_text_splitter()
#     chunk_documents = []
    
#     print(f"Processing {len(documents)} documents into chunks...")
    
#     for doc in documents:
#         doc_chunks = text_chunker.split_into_chunks(doc.page_content, text_splitter)
#         source_url = doc.metadata.get('source', 'unknown')
#         total_chunks = len(doc_chunks)
        
#         for i, chunk_text in enumerate(doc_chunks):
#             chunk_doc = Document(
#                 page_content=chunk_text,
#                 metadata={
#                     'source': source_url,
#                     'chunk_index': i,
#                     **doc.metadata
#                 }
#             )
#             chunk_documents.append(chunk_doc)
            
#             chunk_id = generate_chunk_id(source_url, i)
#             add_chunk(chunk_id, source_url, total_chunks)
    
#     print(f"Creating vector store with {len(chunk_documents)} document chunks...")
    
#     vector_database.create_vector_store(chunk_documents, embeddings_model)
#     print("Vector store created successfully!")
#     print(f"Created {len(chunk_documents)} chunk metadata records")

#     llm = ChatOpenAI(
#         model_name=settings.LLM_MODEL,
#         temperature=settings.TEMPERATURE,
#         openai_api_key=api_key
#     )

#     prompt = PromptTemplate(
#         template=settings.RAG_TEMPLATE,
#         input_variables=["context", "question"]
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_database.create_retriever(embeddings_model, search_kwargs={"k": settings.TOP_K_CHUNKS}),
#         chain_type_kwargs={"prompt": prompt}
#     )
    
#     return qa_chain

def ask(qa_chain, question):
    if not qa_chain:
        raise RuntimeError("RAG system not initialized")
    
    try:
        retriever = qa_chain.retriever
        retrieved_docs = retriever.invoke(question)
        
        source_urls = []
        for doc in retrieved_docs:
            source_url = doc.metadata.get('source', 'unknown')
            if source_url != 'unknown':
                source_urls.append(source_url)
        
        response = qa_chain.invoke({"query": question})
        
        return {
            "answer": response["result"],
            "sources": list(set(source_urls)),
            "source_count": len(set(source_urls))
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "sources": [],
            "source_count": 0
        }
