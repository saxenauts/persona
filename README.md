# Persona Graph - Intelligent User Memory

Persona Graph is an innovative digital identity system that evolves with a user's digital footprint. By creating and maintaining a dynamic knowledge graph for each user, the graph provides contextually rich information to any programming interface, particularly LLM-based systems, enabling next gen personalization and user-centric experiences. At its core, persona graph is an attempt to create a memetic digital organism that can evolve and grow with the user, and represent the user's mindspace digitally. 

## Vision

We believe in a future where digital interactions are seamlessly personalized, respecting user privacy while delivering unparalleled value. Our mission is to empower developers and businesses to create user experiences that are not just tailored, but truly understand and grow with each individual user.

Check the introductory blog post [here](https://open.substack.com/pub/saxenauts/p/memetics-meets-ai-building-your-digital?r=5djzi&utm_campaign=post&utm_medium=web)

![Sample Graph Update](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab907625-6e05-4516-8266-01432ad6240e_1710x1558.gif)

## Features

- Dynamic user knowledge graph construction
- Contextual query processing using RAG (Retrieval-Augmented Generation)
- Scalable and efficient graph storage with Neo4j
- FastAPI backend for high-performance API interactions
- OpenAI integration for advanced natural language processing
- Docker-based deployment for easy setup and scaling

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/saxenauts/persona-graph.git
   cd persona-graph
   ```

2. Ensure you have Docker and Docker Compose installed on your system.

3. Create a `.env` file in the app directory with the following content:
   ```
   NEO4J_URI=neo4j://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_secure_password
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Start the services:
   ```
   docker-compose up -d
   ```

5. The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

## API Usage

### Create a New User

```bash
curl -X POST "http://localhost:8000/api/v1/users" -H "Content-Type: application/json" -d '{"user_id": "alice123"}'
```

### Ingest User Data

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/alice123" -H "Content-Type: application/json" -d '{"content": "Alice is a software engineer who loves hiking and photography."}'
```

### Perform a RAG Query

```bash
curl -X POST "http://localhost:8000/api/v1/rag/alice123/query" -H "Content-Type: application/json" -d '{"query": "What are Alice'\''s hobbies?"}'
```

### Examples

See the [examples.ipynb](examples.ipynb) file for a sample product recommendation use case. 


## Supercharging Personalization: Use Cases

1. **E-commerce Product Recommendations**
   Innernet can analyze a user's browsing history, purchase patterns, and stated preferences to provide hyper-personalized product recommendations that evolve with the user's tastes over time.

2. **Content Streaming Platforms**
   By understanding a user's viewing habits, genre preferences, and even mood patterns, Innernet can help streaming services offer content suggestions that are uncannily accurate and timely.

3. **Personal Finance Apps**
   Innernet can help finance apps provide tailored advice by understanding a user's spending habits, financial goals, and risk tolerance, adapting as the user's financial situation changes.

4. **Health and Fitness Applications**
   By tracking a user's exercise routines, dietary preferences, and health goals, Innernet can assist in providing personalized workout plans and nutrition advice that adapts as the user's fitness journey progresses.

## TODOs

- [ ] Write tests and evals for graph construction
- [ ] Add activation, and forgetting functions to the graph update 
- [ ] Add manual graph update functions to the API
- [ ] Add a graph visualization tool in D3
- [ ] Write notebook for ingesting X posts and bookmarks to the graph
- [ ] Write notebook for ingesting a users browsing history to the graph


## Architecture

Innernet User Memory uses FastAPI for the backend, Neo4j for graph storage, and OpenAI for natural language processing. The entire system is containerized using Docker for easy deployment.

## License

This project is licensed under the MIT License.
