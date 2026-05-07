#!/bin/bash
set -e
echo "Activating virtual environment..."
source .venv/bin/activate

if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
else
    echo "Ollama already running."
fi

echo "Starting backend..."
python -m uvicorn api.main:app --reload &

sleep 3

