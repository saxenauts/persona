# Persona Graph - Intelligent User Memory

## Overview

Persona Graph is an innovative digital identity system that evolves with a user's digital footprint. By creating and maintaining a dynamic knowledge graph for each user, it provides contextually rich information to any programming interface, particularly LLM-based systems, enabling next-generation personalization and user-centric experiences. At its core, Persona Graph aims to create a memetic digital organism that can evolve and grow with the user, representing the user's mindspace digitally.

## Vision

We believe in a future where digital interactions are seamlessly personalized, respecting user privacy while delivering unparalleled value. Our mission is to empower developers and businesses to create user experiences that are not just tailored, but truly understand and grow with each individual user.

Check the introductory blog post [here](https://saxenauts.io/blog/persona-graph)


## Features

- **Dynamic User Knowledge Graph Construction:** Automatically builds and updates a user's knowledge graph based on their interactions and data.
- **Contextual Query Processing using RAG (Retrieval-Augmented Generation):** Enhances query responses by leveraging the user's knowledge graph.
- **Scalable and Efficient Graph Storage with Neo4j:** Utilizes Neo4j for robust and scalable graph database management.
- **FastAPI Backend for High-Performance API Interactions:** Provides a fast and efficient backend for handling API requests.
- **OpenAI Integration for Advanced Natural Language Processing:** Leverages OpenAI's models for sophisticated NLP tasks.
- **Docker-Based Deployment for Easy Setup and Scaling:** Ensures seamless deployment and scaling using Docker containers.

## Installation and Setup

### Prerequisites

- **Docker:** Ensure Docker is installed on your system. [Download Docker](https://www.docker.com/get-started)
- **Docker Compose:** Ensure Docker Compose is installed. [Install Docker Compose](https://docs.docker.com/compose/install/)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saxenauts/persona-graph.git
   cd persona-graph
   ```

2. **Set Up Environment Variables:**

   Create a `.env` file in the `app` directory with the following content:

   ```env
   NEO4J_URI=neo4j://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_secure_password
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Start the Services:**

   ```bash
   docker-compose up -d
   ```

4. **Access the API:**

   The API will be available at `http://localhost:8000`. Access the API documentation at `http://localhost:8000/docs`.

## Architecture

Persona Graph leverages the following technologies:

- **FastAPI:** Serves as the backend framework, facilitating high-performance API interactions.
- **Neo4j:** Manages the graph database, storing user knowledge graphs efficiently.
- **OpenAI:** Provides advanced natural language processing capabilities.
- **Docker:** Ensures the application is containerized for easy deployment and scalability.

### Component Interaction

1. **User Interaction:** Users interact with the system through RESTful APIs.
2. **Data Ingestion:** Unstructured data is ingested and processed to extract meaningful entities.
3. **Knowledge Graph Construction:** Extracted entities are used to build and update the user's knowledge graph in Neo4j.
4. **Query Processing:** User queries are processed using Retrieval-Augmented Generation (RAG), leveraging the knowledge graph for contextually rich responses.
5. **Response Generation:** OpenAI models generate responses based on the contextual information retrieved from the knowledge graph.

## API Documentation

Detailed API documentation is available at `http://localhost:8000/docs` once the services are up and running.

### Endpoints

#### 1. Create a New User

- **URL:** `/api/v1/users`
- **Method:** `POST`
- **Body:**
  
  ```json
  {
    "user_id": "alice123"
  }
  ```

- **Response:**

  ```json
  {
    "message": "User alice123 created successfully"
  }
  ```

#### 2. Ingest User Data

- **URL:** `/api/v1/ingest/{user_id}`
- **Method:** `POST`
- **Body:**
  
  ```json
  {
    "content": "Alice is a software engineer who loves hiking and photography."
  }
  ```

- **Response:**

  ```json
  {
    "message": "Data ingested successfully"
  }
  ```

#### 3. Perform a RAG Query

- **URL:** `/api/v1/rag/{user_id}/query`
- **Method:** `POST`
- **Body:**
  
  ```json
  {
    "query": "What are Alice's hobbies?"
  }
  ```

- **Response:**

  ```json
  {
    "answer": "Alice enjoys hiking and photography."
  }
  ```

#### 4. Delete a User

- **URL:** `/api/v1/users/{user_id}`
- **Method:** `DELETE`
- **Response:**

  ```json
  {
    "message": "User alice123 deleted successfully"
  }
  ```

#### 5. Test Constructor Flow

- **URL:** `/api/v1/test-constructor-flow`
- **Method:** `POST`
- **Response:**

  ```json
  {
    "status": "Graph updated successfully",
    "context": "..."
  }
  ```

### Models

#### UserCreate

- **Fields:**
  - `user_id` (string): Unique identifier for the user.

#### IngestData

- **Fields:**
  - `content` (string): The content to ingest into the user's knowledge graph.

#### RAGQuery

- **Fields:**
  - `query` (string): The user's query.

#### RAGResponse

- **Fields:**
  - `answer` (string): The response generated based on the user's query.

## Developer Guide

### Setup for Development

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saxenauts/persona-graph.git
   cd persona-graph
   ```

2. **Install Dependencies:**

   Ensure you have [Poetry](https://python-poetry.org/docs/) installed.

   ```bash
   poetry install
   ```

3. **Set Up Environment Variables:**

   Create a `.env` file in the `app` directory with necessary configurations as described in the [Installation and Setup](#installation-and-setup) section.

4. **Run Tests:**

   ```bash
   poetry run pytest
   ```

### Code Structure

- **app/**: Contains the main application code.
  - **graph/**: Handles graph database operations and construction.
  - **openai/**: Manages interactions with OpenAI's APIs.
  - **routers/**: Defines API endpoints.
  - **utils/**: Utility functions and data models.
  - **api/**: Service layers for different functionalities.
- **tests/**: Contains all test cases.
- **docs/**: Documentation files.
- **docker-compose.yml**: Docker configuration for setting up services.
- **Dockerfile**: Dockerfile for building the application container.
- **pyproject.toml**: Poetry configuration file.
- **README.md**: Project overview and setup instructions.

### Extending the Application

1. **Adding New Features:**
   - Follow the existing patterns for adding new services, models, and API endpoints.
   - Ensure that all new functionalities are tested appropriately.

2. **Switching Graph Databases:**
   - Abstract the graph database operations to allow easy switching between different graph databases.
   - Implement interface layers that can interact with multiple graph databases seamlessly.

3. **Integrating Different LLMs:**
   - Abstract the LLM interactions to support multiple providers.
   - Implement factory patterns or dependency injection to manage different LLM integrations.


## Testing

## License

This project is licensed under the MIT License.
