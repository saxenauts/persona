from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from .models import Message, Conversation
from .processors import ChatProcessor

class ChatStorageInterface(ABC):
    @abstractmethod
    async def store_message(self, user_id: str, conversation_id: str, message: Message) -> bool:
        """Store a single message in a conversation"""
        pass

    @abstractmethod
    async def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a complete conversation"""
        pass

    @abstractmethod
    async def get_recent_messages(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Message]:
        """Get most recent messages from a conversation"""
        pass

    @abstractmethod
    async def get_messages_by_timerange(self, user_id: str, conversation_id: str, 
                                      start_time: datetime, end_time: datetime) -> List[Message]:
        """Get messages within a time range"""
        pass

    @abstractmethod
    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation and return its ID"""
        pass

class ChatAPI:
    """High-level API for chat operations"""
    def __init__(self, 
                 storage_backend: str = "neo4j",
                 connection_string: str = None,
                 engine = None,
                 schema: str = "luna_chat"):
        
        if storage_backend == "neo4j":
            from .neo4j_handler import Neo4jChatStorage
            self.storage = Neo4jChatStorage()
        elif storage_backend == "postgres":
            from .postgres_handler import PostgresChatStorage
            self.storage = PostgresChatStorage(
                connection_string=connection_string,
                engine=engine,
                schema=schema
            )
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        
        self.processor = ChatProcessor(self.storage)

    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation"""
        return await self.storage.create_conversation(user_id, metadata)

    async def add_message(self, user_id: str, conversation_id: str, 
                         role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to a conversation"""
        message = Message(role=role, content=content, metadata=metadata or {})
        return await self.processor.add_message(user_id, conversation_id, message)

    async def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Conversation]:
        """Get a complete conversation"""
        return await self.storage.get_conversation(user_id, conversation_id)

    async def get_recent_messages(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Message]:
        """Get most recent messages"""
        return await self.storage.get_recent_messages(user_id, conversation_id, limit)

    async def process_conversation(self, user_id: str, conversation_id: str, 
                                 max_tokens: Optional[int] = None) -> Optional[Conversation]:
        """Process and truncate conversation"""
        conv = await self.get_conversation(user_id, conversation_id)
        if conv:
            return await self.processor.truncate_conversation(conv, max_tokens)
        return None 