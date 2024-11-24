# Luna9 - Intelligent User Memory

![Luna9 Banner](docs/assets/banner.svg)

## Overview

Luna is a language dependent digital identity system that evolves with a user's digital footprint.

It creates and maintains a dynamic knowledge graph for each user, it provides contextually rich information to any programming interface, particularly LLM-based systems, enabling next-generation personalization and user-centric experiences.

At its core, Luna aims to create a memetic digital organism that can evolve and grow with the user, representing the user's mindspace digitally.

## Vision 

Digital future is getting more personalized, and our apps and our web are getting data rich. The interfaces should naturally evolve and allow more personalized and dynamic experiences. Personalization at it's fundamental is about understanding the user.

Traditionally we have been understanding user as fixed and static relational data points in tables or vector embeddings. But our minds and identities are associative and dynamic in nature.

We use language to connect different parts of our lives. With LLMs it's possible to map this complexity and to make sense of it. So our user representation should not be left behind. 

Read the motivation and design decisions in depth [here](https://saxenauts.io/blog/persona-graph)

Read the docs [here](http://docs.luna9.dev), see the example of transforming a simple app and supercharge it with personalization. 

Luna is designed to build graph from unstructured user data like interaction logs, emails, chats, etc. 

While Luna can support storing conversational history, it's not the primary purpose. There are better tools for that like Mem0.ai, MemGPT, etc.


## Features

- **Dynamic User Knowledge Graph Construction:** Automatically builds and updates a user's knowledge graph based on their interactions data.
- **Custom Knowledge API:** Provide custom schema specific to your application and learn that information from your app's interaction logs. 
- **Contextual Query Processing using RAG:** Enhances query responses by leveraging the user's knowledge graph.


## Installation and Setup

### Prerequisites

- **Docker:** Ensure Docker is installed on your system. [Download Docker](https://www.docker.com/get-started)
- **Docker Compose:** Ensure Docker Compose is installed. [Install Docker Compose](https://docs.docker.com/compose/install/)

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saxenauts/luna9.git
   cd luna9
   ```

2. **Set Up Environment Variables:**

   Use `.env.example` to create a `.env` file in the root directory with the following content:

   ```env
   URI_NEO4J=neo4j://neo4j:7687
   USER_NEO4J=neo4j
   PASSWORD_NEO4J=your_secure_password

   OPENAI_API_KEY=your_openai_api_key

   NEO4J_AUTH=neo4j/your_secure_password
   ```

3. **Start the Services:**

   ```bash
   docker-compose up -d
   ```

4. **Access the API:**

   The API will be available at `http://localhost:8000`. Access the API documentation at `http://localhost:8000/docs`.
   Check swagger UI at `http://localhost:8000/docs` to go through the endpoints.

## Architecture

Luna9 has the following components:

- **GraphOps:** Abstraction layer for graph database operations. Currently supports Neo4j but extensible.
- **Constructor:** Constructs the user's knowledge graph.
- **GraphContextRetriever:** Functions to fetch the relevant context from user's graph.
- **LLMGraph:** All OpenAI API calls in Luna for graph construction, community detection, etc.
- **Services:** Business logic for ingesting data, creating communities, rag, learning, asking services, and for adding custom data to graph. 
- **API:** FastAPI server to serve the API endpoints. Easy to extend with new functionalities.

Luna9 uses following technologies:

- **Neo4j:** Graph database to store user's knowledge graph.
- **Neo4j:** For vector database, we use HNSW. Neo4j uses Apache Lucene for their vector Index. 
- **OpenAI:** All LLM calls.
- **FastAPI:** API server.
- **Docker:** Containerization for easy deployment and scalability.

We plan to add LLM, Graph & Vector DB abstractions to extend these functionalities to other tools and frameworks.


## API Documentation

Detailed API documentation is available at `http://localhost:8000/docs` once the services are up and running.

Have a look at [docs](http://docs.luna9.dev) for examples and API usage. 


### Code Structure

- **app_server/**: FastAPI server code.
- **persona_graph/**: Contains the main application code.
  - **core/**: Core functionalities and database migrations.
  - **llm/**: All LLM calls in Luna.
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
