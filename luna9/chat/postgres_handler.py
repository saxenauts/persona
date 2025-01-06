from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import Table, Column, String, DateTime, JSON, MetaData, ForeignKey, select
from .interface import ChatStorageInterface
from .models import Message, Conversation, Role

class PostgresChatStorage(ChatStorageInterface):
    def __init__(self, connection_string: str = None, engine = None, schema: str = "luna_chat"):
        """Initialize PostgreSQL storage"""
        if engine:
            self.engine = engine
        elif connection_string:
            self.engine = create_async_engine(connection_string)
        else:
            raise ValueError("Either connection_string or engine must be provided")

        self.schema = schema
        self.metadata = MetaData(schema=schema)
        
        # Define tables
        self.conversations = Table(
            'conversations', self.metadata,
            Column('id', String, primary_key=True),
            Column('user_id', String, index=True),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('metadata', JSON)
        )

        self.messages = Table(
            'messages', self.metadata,
            Column('id', String, primary_key=True),
            Column('conversation_id', String, ForeignKey('conversations.id')),
            Column('role', String),
            Column('content', String),
            Column('timestamp', DateTime),
            Column('metadata', JSON)
        )

    async def init_schema(self):
        """Create schema and tables if they don't exist"""
        async with self.engine.begin() as conn:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS {self.schema}')
            await conn.run_sync(self.metadata.create_all)

    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        async with self.engine.begin() as conn:
            await conn.execute(
                self.conversations.insert().values(
                    id=conversation_id,
                    user_id=user_id,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {}
                )
            )
        return conversation_id

    async def store_message(self, user_id: str, conversation_id: str, message: Message) -> bool:
        """Store a single message in a conversation"""
        try:
            async with self.engine.begin() as conn:
                # First verify the conversation belongs to the user
                conv_query = select(self.conversations).where(
                    self.conversations.c.id == conversation_id,
                    self.conversations.c.user_id == user_id
                )
                result = await conn.execute(conv_query)
                if not result.first():
                    return False

                # Insert the message
                await conn.execute(
                    self.messages.insert().values(
                        id=str(uuid.uuid4()),
                        conversation_id=conversation_id,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp,
                        metadata=message.metadata
                    )
                )

                # Update conversation's updated_at timestamp
                await conn.execute(
                    self.conversations.update()
                    .where(self.conversations.c.id == conversation_id)
                    .values(updated_at=datetime.utcnow())
                )
                return True
        except Exception as e:
            print(f"Error storing message: {e}")
            return False

    async def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a complete conversation"""
        try:
            async with self.engine.begin() as conn:
                # Get conversation details
                conv_query = select(self.conversations).where(
                    self.conversations.c.id == conversation_id,
                    self.conversations.c.user_id == user_id
                )
                conv_result = await conn.execute(conv_query)
                conv_row = conv_result.first()
                
                if not conv_row:
                    return None

                # Get all messages for the conversation
                msg_query = select(self.messages).where(
                    self.messages.c.conversation_id == conversation_id
                ).order_by(self.messages.c.timestamp)
                
                msg_result = await conn.execute(msg_query)
                messages = []
                
                for row in msg_result:
                    messages.append(Message(
                        role=Role(row.role),
                        content=row.content,
                        timestamp=row.timestamp,
                        metadata=row.metadata
                    ))

                return Conversation(
                    id=conv_row.id,
                    user_id=conv_row.user_id,
                    messages=messages,
                    created_at=conv_row.created_at,
                    updated_at=conv_row.updated_at,
                    metadata=conv_row.metadata
                )
        except Exception as e:
            print(f"Error retrieving conversation: {e}")
            return None

    async def get_recent_messages(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Message]:
        """Get most recent messages from a conversation"""
        try:
            async with self.engine.begin() as conn:
                # Verify conversation belongs to user
                conv_query = select(self.conversations).where(
                    self.conversations.c.id == conversation_id,
                    self.conversations.c.user_id == user_id
                )
                conv_result = await conn.execute(conv_query)
                if not conv_result.first():
                    return []

                # Get recent messages
                query = select(self.messages).where(
                    self.messages.c.conversation_id == conversation_id
                ).order_by(
                    self.messages.c.timestamp.desc()
                ).limit(limit)

                result = await conn.execute(query)
                messages = []
                
                for row in result:
                    messages.append(Message(
                        role=Role(row.role),
                        content=row.content,
                        timestamp=row.timestamp,
                        metadata=row.metadata
                    ))
                
                messages.reverse()  # Restore chronological order
                return messages
        except Exception as e:
            print(f"Error retrieving recent messages: {e}")
            return []

    async def get_messages_by_timerange(
        self,
        user_id: str,
        conversation_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Message]:
        """Get messages within a specific time range"""
        try:
            async with self.engine.begin() as conn:
                # Verify conversation belongs to user
                conv_query = select(self.conversations).where(
                    self.conversations.c.id == conversation_id,
                    self.conversations.c.user_id == user_id
                )
                conv_result = await conn.execute(conv_query)
                if not conv_result.first():
                    return []

                # Get messages within timerange
                query = select(self.messages).where(
                    self.messages.c.conversation_id == conversation_id,
                    self.messages.c.timestamp >= start_time,
                    self.messages.c.timestamp <= end_time
                ).order_by(self.messages.c.timestamp)

                result = await conn.execute(query)
                return [
                    Message(
                        role=Role(row.role),
                        content=row.content,
                        timestamp=row.timestamp,
                        metadata=row.metadata
                    )
                    for row in result
                ]
        except Exception as e:
            print(f"Error retrieving messages by timerange: {e}")
            return [] 