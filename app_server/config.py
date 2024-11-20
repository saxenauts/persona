from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from os import environ
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Info(BaseModel):
    """Information about the API"""
    title: str = Field("Innernet API", description="API title")
    description: str = Field("Backend API for Innernet", description="API description")
    version: str = Field("0.1.0", description="API version")
    root_path: str = Field("/", description="API root path")
    docs_url: Optional[str] = Field("/docs", description="API documentation URL")
    redoc_url: Optional[str] = Field("/redoc", description="ReDoc documentation URL")
    swagger_ui_parameters: dict = Field(
        {"displayRequestDuration": True},
        description="Swagger UI parameters"
    )

class Database(BaseModel):
    """Database configuration"""
    HOSTNAME: str = Field(environ.get("DB_HOST", "graph-db"), description="Database hostname")
    PORT: int = Field(environ.get("DB_PORT", 5432), description="Database port")
    USER: str = Field(environ.get("DB_USER", "postgres"), description="Database username")
    PASSWORD: str = Field(environ.get("DB_PASSWORD", "postgres"), description="Database password")
    NAME: str = Field(environ.get("DB_NAME", "color"), description="Database name")

    @property
    def URI(self) -> str:
        """Database URI for asyncpg"""
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOSTNAME}:{self.PORT}/{self.NAME}"

    @property
    def URI_LLM(self) -> str:
        """Database URI for LLM"""
        return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOSTNAME}:{self.PORT}/{self.NAME}"

class Neo4j(BaseModel):
    """Neo4j configuration"""
    URI: str = Field(environ.get("NEO4J_URI", "neo4j://neo4j:7687"), description="Neo4j URI")
    USER: str = Field("neo4j", description="Neo4j username")
    PASSWORD: str = Field("passwordz", description="Neo4j password")

class ML(BaseModel):
    """Machine Learning configuration"""
    URI: str = Field(environ.get("ML_URI", "http://color-ml-local:8080"), description="ML service URI")
    KW_EXTRACTION_MODEL_NAME: str = Field(
        environ.get("ML_KW_EXTRACTION_MODEL_NAME", "keybert"),
        description="Keyword extraction model name"
    )
    EMBEDDING_MODEL_NAME: str = Field(
        environ.get("ML_EMBEDDING_MODEL_NAME", "miniml"),
        description="Embedding model name"
    )
    OPENAI_KEY: str = Field(environ.get("OPENAI_KEY", ""), description="OpenAI API key")
    OPENAI_ORG: str = Field(environ.get("OPENAI_ORG", ""), description="OpenAI organization")
    OPENAI_TEXT_COMPLETION_MODEL: str = Field("gpt-3.5-turbo", description="OpenAI text completion model")

class BaseConfig(BaseSettings):
    """Base configuration for the application"""
    INFO: Info = Info()
    DB: Database = Database()
    NEO4J: Neo4j = Neo4j()
    MACHINE_LEARNING: ML = ML()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = BaseConfig()