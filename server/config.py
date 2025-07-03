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
    OPENAI_API_KEY: str = Field(environ.get("OPENAI_API_KEY", ""), description="OpenAI API key")

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

config = BaseConfig()
