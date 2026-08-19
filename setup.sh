#!/bin/bash
set -e

echo "Downloading SpaCy English model..."
python -m spacy download en_core_web_md

echo "Pulling Ollama Qwen2.5:7b-instruct model..."
ollama pull qwen2.5:7b-instruct

echo "Setup completed successfully!"
