# DS4300-LLM
Practical 2 for DS4300 Northeastern University

Authors: Grace Kelner, Milena Perez-Gurus, Claire Mahon

## Setup

There are numerous installations that need to happen to ensure you can test all the pipelines we have setup within the "Pipelines" folder.
- Ollama: Ensure you download Ollama from online and follow documentation for setup.
  - Once ollama is installed, run "ollama pull llama3.2" and "ollama pull mistral" in your terminal to ensure you have both the LLM models needed.
- Redis: Install on docker.
- Chroma: 
- Weaviate: Install weaviate from online and follow documentation for setup: https://weaviate.io/developers/weaviate
- Nomic: Run "ollama pull nomic" in your terminal
- Mpnet-base-v2:
- MiniLM:

  ## How to run

You will want to enter the Pipelines folder, each model is tested in a separate script.  Within the script, we automatically return the time it takes to return each response, as well as the time it takes to ingest and the memory it takes to store.
