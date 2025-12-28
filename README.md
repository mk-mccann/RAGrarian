---
title: Permacore
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.2.0  
python_version: "3.13"
app_file: permacore/app.py
pinned: true
---

# Permacore 🌿 Chatbot for Permaculture, Regenerative Agriculture, and Sustainable Developent

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot to provide accurate, context-aware answers about sustainable farming practices, ecological design principles, and regenerative systems.

This project uses MistralAI for the LLM backend, LangChain to implement the RAG system, and Gradio for the UI.

All data sources are either open-source or usage permissions have been granted by the respective publishers.

## Goals

- Provide accessible information about permaculture principles and practices
- Support learning and decision-making in regenerative agriculture
- Offer evidence-based guidance on sustainable development techniques
- Enable natural language queries about complex ecological topics

## Features

- Context-aware responses and source citations using RAG architecture
- Knowledge base covering permaculture design, soil health, water management, and more
- Conversational interface for exploring sustainable agriculture topics

## Demo on Hugging Face

Run an interactive demo at [https://huggingface.co/spaces/mk-mccann/Permacore](https://huggingface.co/spaces/mk-mccann/Permacore)

## Installation

A `conda` environment is recommended for installation. Then, 

```bash
pip install -r requirements.txt
```

## Interactive UI

The UI is powered by Gradio. A local UI can be started by running `python app.py`.

## Command Line Usage

```python
# Example usage
from os import getenv
from dotenv import load_dotenv
from permacore.rag_agent import RAGAgent
from config import ChromaConfig, LLMConfig, RetrievalConfig

load_dotenv()
mistral_api_key = getenv("MISTRAL_API_KEY")

# Set up configurations
chroma_config = ChromaConfig()
chroma_config.embeddings = MistralAIEmbeddings(api_key=mistral_api_key)    # type: ignore
llm_config = LLMConfig()
retrieval_config = RetrievalConfig()

# Initialize the RAG agent
agent = RAGAgent(
    chroma_config=chroma_config,
    llm_config=llm_config,
    retrieval_config=retrieval_config,
)

# For a single query:
response = agent.query("What are the three ethics of permaculture?")
print(response)

# For interactive chat:
agent.cli_chat()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

Suggestions for resources to share are also welcome!

## Data Sources

- [Open Source Ecology](https://wiki.opensourceecology.org/wiki/)
- [One Commuity Global](http://onecommunityglobal.org/)

## License

GPL-3.0
