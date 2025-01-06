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
    @classmethod
    async def create(cls, 
                    storage_backend: str = "sqlite",
                    connection_string: str = None,
                    engine = None,
                    schema: str = "luna_chat",
                    db_path: str = "chat.db"):
        """Factory method to create and initialize ChatAPI"""
        self = cls()

        print(f"Creating ChatAPI with storage backend: {storage_backend}")
        if storage_backend == "postgres":
            from .postgres_handler import PostgresChatStorage
            self.storage = PostgresChatStorage(
                connection_string=connection_string,
                engine=engine,
                schema=schema
            )
            await self.storage.init_schema()
        elif storage_backend == "sqlite":
            from .sqlite_handler import SqliteChatStorage
            self.storage = SqliteChatStorage(db_path=db_path)
            await self.storage.init_schema()
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        
        self.processor = ChatProcessor(self.storage)
        return self

    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation"""
        return await self.storage.create_conversation(user_id, metadata)

    async def add_message(self, user_id: str, conversation_id: str, 
                         role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to a conversation"""
        message = Message(
            role=role, 
            content=content, 
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
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