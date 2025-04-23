#!/bin/bash

# Run nay setup steps or pre-processing task here
echo "Starting article RAG FastAPI service..."

# Start the main application
uvicorn chatbot_api.main:app --host 0.0.0.0 --port 8000
