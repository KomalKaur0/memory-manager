#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Activate virtual environment and start server
source .venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload