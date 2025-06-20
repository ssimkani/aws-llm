#!/bin/sh

if ! ollama list | grep -q llama3; then
  ollama pull llama3.2:latest
fi
ollama serve