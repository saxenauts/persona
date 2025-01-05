from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from persona_graph.chat.interface import ChatAPI
from persona_graph.chat.models import Message, Conversation
from pydantic import BaseModel

router = APIRouter(prefix="/chat")
chat_api = ChatAPI()

class CreateConversationRequest(BaseModel):
    user_id: str
    metadata: Optional[Dict[str, Any]] = None

class AddMessageRequest(BaseModel):
    user_id: str
    conversation_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/conversations")
async def create_conversation(request: CreateConversationRequest):
    conversation_id = await chat_api.create_conversation(request.user_id, request.metadata)
    return {"conversation_id": conversation_id}

@router.post("/messages")
async def add_message(request: AddMessageRequest):
    success = await chat_api.add_message(
        request.user_id,
        request.conversation_id,
        request.role,
        request.content,
        request.metadata
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add message")
    return {"status": "success"}

@router.get("/conversations/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    conversation = await chat_api.get_conversation(user_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.get("/conversations/{user_id}/{conversation_id}/recent")
async def get_recent_messages(user_id: str, conversation_id: str, limit: int = 10):
    messages = await chat_api.get_recent_messages(user_id, conversation_id, limit)
    return {"messages": messages}