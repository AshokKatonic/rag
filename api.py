from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag_engine import setup_rag_system, ask
from src.auth import verify_token, create_access_token
import uuid
import time
import json
from datetime import timedelta, datetime


app = FastAPI(
    title="RAG API",
    description="API for RAG with JWT Authentication",
    version="1.0.0"
)

rag = setup_rag_system()
if rag is None:
    print("Warning: RAG system not initialized. Please run 'python reload_rag.py' to create the vector store.") 

class Message(BaseModel):
    role: str  
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False

class TokenRequest(BaseModel):
    client_name: str = "default_client"
    expires_hours: int = 168  # 7 days default

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
    client: str

class ChatResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatResponseChoice]
    sources: list[str] = None
    source_count: int = None

@app.get("/")
def root():
    return {
        "message": "RAG API with JWT Authentication",
        "version": "1.0.0",
        "endpoints": {
            "generate_token": "/auth/token",
            "chat": "/v1/chat/completions (requires JWT token)",
            "docs": "/docs"
        }
    }

@app.post("/auth/token", response_model=TokenResponse)
def generate_token(request: TokenRequest):

    try:
        if request.expires_hours > 720:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum token expiration is 720 hours (30 days)"
            )
        
        if request.expires_hours < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Minimum token expiration is 1 hour"
            )
        
        # Create token with custom expiration
        token_data = {
            "client": request.client_name,
            "created_at": datetime.utcnow().isoformat()
        }
        
        expires_delta = timedelta(hours=request.expires_hours)
        access_token = create_access_token(token_data, expires_delta)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in_hours=request.expires_hours,
            client=request.client_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating token: {str(e)}"
        )

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest, token_data: dict = Depends(verify_token)):

    if rag is None:
        return {"error": "RAG system not initialized. Please run 'python reload_rag.py' to create the vector store."}
    
    try:
        user_message = next((m.content for m in request.messages if m.role == "user"), "")
        if not user_message:
            return {"error": "No user message found in request"}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}
    
    if request.stream:
        def generate():
            response_data = ask(rag, user_message)
            answer = response_data["answer"]
            sources = response_data.get("sources", [])

            for token in answer.split():
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token + " "},
                        "finish_reason": None
                    }]
                }

                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.05)
            final_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            if sources:
                sources_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"sources": sources, "source_count": len(sources)},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(sources_chunk)}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )

    response_data = ask(rag, user_message)
    answer = response_data["answer"]
    sources = response_data.get("sources", [])
    
    response = ChatResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatResponseChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        sources=sources,
        source_count=len(sources)
    )
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)