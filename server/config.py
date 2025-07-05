from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from os import environ
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Info(BaseModel):
    """Information about the API"""
    title: str = Field("Persona API", description="API title")
    description: str = Field("Backend API for Persona", description="API description")
    version: str = Field("1.0.0", description="API version")
    root_path: str = Field("/", description="API root path")
    docs_url: Optional[str] = Field("/docs", description="API documentation URL")
    redoc_url: Optional[str] = Field("/redoc", description="ReDoc documentation URL")
    swagger_ui_parameters: dict = Field(
        {"displayRequestDuration": True},
        description="Swagger UI parameters"
    )


class Neo4j(BaseModel):
    """Neo4j configuration"""
    URI: str = Field(environ.get("URI_NEO4J", ""), description="Neo4j URI")
    USER: str = Field(environ.get("USER_NEO4J", ""), description="Neo4j username")
    PASSWORD: str = Field(environ.get("PASSWORD_NEO4J", ""), description="Neo4j password")

class ML(BaseModel):
    """Machine Learning configuration"""
    # LLM Service Configuration
    LLM_SERVICE: str = Field(environ.get("LLM_SERVICE", "openai/gpt-4o-mini"), description="LLM service in format 'provider/model'")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(environ.get("OPENAI_API_KEY", ""), description="OpenAI API key")
    OPENAI_CHAT_MODEL: str = Field(environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"), description="OpenAI chat model")
    OPENAI_EMBEDDING_MODEL: str = Field(environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), description="OpenAI embedding model")
    
    # Azure OpenAI Configuration
    AZURE_API_KEY: str = Field(environ.get("AZURE_API_KEY", ""), description="Azure OpenAI API key")
    AZURE_API_BASE: str = Field(environ.get("AZURE_API_BASE", ""), description="Azure OpenAI API base URL")
    AZURE_API_VERSION: str = Field(environ.get("AZURE_API_VERSION", "2024-02-01"), description="Azure OpenAI API version")
    AZURE_CHAT_DEPLOYMENT: str = Field(environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"), description="Azure chat model deployment name")
    AZURE_EMBEDDING_DEPLOYMENT: str = Field(environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"), description="Azure embedding model deployment name")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: str = Field(environ.get("ANTHROPIC_API_KEY", ""), description="Anthropic API key")
    ANTHROPIC_CHAT_MODEL: str = Field(environ.get("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-20241022"), description="Anthropic chat model")
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = Field(environ.get("GEMINI_API_KEY", ""), description="Google Gemini API key")
    GEMINI_CHAT_MODEL: str = Field(environ.get("GEMINI_CHAT_MODEL", "gemini-1.5-flash"), description="Google Gemini chat model")

class BaseConfig(BaseSettings):
    """
    Defines the application's configuration settings.
    Utilizes pydantic-settings to automatically read from environment variables
    or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # General settings
    app_name: str = "Persona"
    INFO: Info = Info()
    NEO4J: Neo4j = Neo4j()
    MACHINE_LEARNING: ML = ML()

# Create a global config instance
config = BaseConfig()
