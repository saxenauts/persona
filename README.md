# Persona - Intelligent User Memory

![Persona Banner](docs/assets/banner.svg)

## Overview

Persona is a language dependent digital identity system that evolves with a user's digital footprint.

It creates and maintains a dynamic knowledge graph for each user, it provides contextually rich information to any programming interface, particularly LLM-based systems, enabling next-generation personalization and user-centric experiences.

At its core, Persona aims to create a memetic digital organism that can evolve and grow with the user, representing the user's mindspace digitally.

## Vision 

Digital future is getting more personalized, and our apps and our web are getting data rich. The interfaces should naturally evolve and allow more personalized and dynamic experiences. Personalization at it's fundamental is about understanding the user.

Traditionally we have been understanding user as fixed and static relational data points in tables or vector embeddings. But our minds and identities are associative and dynamic in nature.

We use language to connect different parts of our lives. With LLMs it's possible to map this complexity and to make sense of it. So our user representation should not be left behind. 

Read the motivation and design decisions in depth [here](https://saxenauts.io/blog/persona-graph)

Read the docs [here](http://docs.buildpersona.ai), see the example of transforming a simple app and supercharge it with personalization. 

Persona is designed to build graph from unstructured user data like interaction logs, emails, chats, etc. 

While Persona can support storing conversational history, it's not the primary purpose. There are better tools for that like Mem0.ai, MemGPT, etc.


## Features

- **Dynamic User Knowledge Graph Construction:** Automatically builds and updates a user's knowledge graph based on their interactions data.
- **Custom Knowledge API:** Provide custom schema specific to your application and learn that information from your app's interaction logs. 
- **Contextual Query Processing using RAG:** Enhances query responses by leveraging the user's knowledge graph.


## Prerequisites

- **Docker & Docker Compose**: Required for running Neo4j and the application
- **OpenAI API Key**: Required for LLM operations (will consume API calls during testing)
- **Python 3.12+**: For local development (optional, for running examples locally)

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saxenauts/persona.git
   cd persona
   ```

2. **Set Up Environment Variables:**

   Create a `.env` file in the root directory with the following content:

   ```env
   URI_NEO4J=neo4j://neo4j:7687
   USER_NEO4J=neo4j
   PASSWORD_NEO4J=your_secure_password

   OPENAI_API_KEY=your_openai_api_key

   NEO4J_AUTH=neo4j/your_secure_password
   ```

   **Important**: Replace `your_openai_api_key` with a valid OpenAI API key. The application will consume API calls for LLM operations.

3. **Start the Services:**

   ```bash
   docker compose up -d
   ```

4. **Access the API:**

   The API will be available at `http://localhost:8000`. Access the API documentation at `http://localhost:8000/docs`.
   Check swagger UI at `http://localhost:8000/docs` to go through the endpoints.

5. **Run Tests**

   Current recommended method to run tests is through container, as the entire setup is container dependant with Neo4j. 
   Test container is part of the compose group, so tests are already run through 'docker compose up' command.
   To run tests separately:

   ```bash
   docker compose run --rm test
   ```
   
   **Note**: Tests will consume OpenAI API calls. Monitor your usage in the OpenAI dashboard.

6. **Try the Example:**

   Run the conversation example to see Persona in action:

   ```bash
   docker compose run --rm -e DOCKER_ENV=1 app python examples/conversation.py
   ```

   This will ingest a sample conversation and build a knowledge graph. You can then explore the results in Neo4j Browser at `http://localhost:7474`.

## Architecture

Persona has the following components:

- **GraphOps:** Abstraction layer for graph database operations. Currently supports Neo4j but extensible.
- **Constructor:** Constructs the user's knowledge graph.
- **GraphContextRetriever:** Functions to fetch the relevant context from user's graph.
- **LLMGraph:** All OpenAI API calls in Persona for graph construction, community detection, etc.
- **Services:** Business logic for ingesting data, creating communities, rag, learning, asking services, and for adding custom data to graph. 
- **API:** FastAPI server to serve the API endpoints. Easy to extend with new functionalities.

Persona uses following technologies:

- **Neo4j:** Graph database to store user's knowledge graph.
- **Neo4j:** For vector database, we use HNSW. Neo4j uses Apache Lucene for their vector Index. 
- **OpenAI:** All LLM calls.
- **FastAPI:** API server.
- **Docker:** Containerization for easy deployment and scalability.

We plan to add LLM, Graph & Vector DB abstractions to extend these functionalities to other tools and frameworks.


## API Documentation

Detailed API documentation is available at `http://localhost:8000/docs` once the services are up and running.

The API follows RESTful patterns with the following endpoints:

- `POST /api/v1/users/{user_id}` - Create a new user
- `DELETE /api/v1/users/{user_id}` - Delete a user
- `POST /api/v1/users/{user_id}/ingest` - Ingest data for a user
- `POST /api/v1/users/{user_id}/rag/query` - Query user's knowledge graph
- `POST /api/v1/users/{user_id}/ask` - Ask structured insights from user's data
- `POST /api/v1/users/{user_id}/custom-data` - Add custom structured data

Have a look at [docs](http://docs.buildpersona.ai) for examples and API usage. 


### Code Structure

- **server/**: FastAPI server code.
- **persona/**: Contains the main application code.
  - **core/**: Core functionalities and database migrations.
  - **llm/**: All LLM calls in Persona.
  - **models/**: Pydantic models for the application.
  - **services/**: Business logic for different functionalities.
  - **utils/**: Utility functions.
- **tests/**: Contains all test cases.
- **docs/**: Documentation files.
- **docker-compose.yml**: Docker configuration for setting up services.
- **Dockerfile**: Dockerfile for building the application container.
- **pyproject.toml**: Poetry configuration file.
- **README.md**: Project overview and setup instructions.


## Roadmap

- [ ] Add LLM, Graph & Vector DB Abstractions for custom services. 
- [ ] Text2Cypher with Graph Schema, and multiple schema support. 
- [ ] User BYOA: Text2Graph2RecSys
- [ ] Quantifiable Functions for Graph
- [ ] Multiple word phrases as memetic units 
- [ ] Self Organizing and Self Growing Graph Agent with Forgetting Mechanism

## License

This project is licensed under the MIT License.
