FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install poetry
RUN pip install poetry

WORKDIR /app

# Copy only the dependency files first
COPY pyproject.toml poetry.lock ./

# Copy the persona package and server
COPY persona ./persona/
COPY server/ ./server/

# Install dependencies and the package
RUN poetry config virtualenvs.create false \
    && poetry install \
    && pip install uvicorn

# Command to run the FastAPI server
CMD ["python", "-m", "server.run"]
