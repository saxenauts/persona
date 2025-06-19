FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install poetry
RUN pip install poetry

WORKDIR /app

# Copy only the dependency files first
COPY pyproject.toml poetry.lock ./

# Copy the persona package, server, and examples
COPY persona ./persona/
COPY server/ ./server/
COPY examples/ ./examples/

# Install dependencies and the package
RUN poetry config virtualenvs.create false \
    && poetry install \
    && pip install uvicorn

# Command to run the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]