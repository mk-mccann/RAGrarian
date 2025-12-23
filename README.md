---
title: Permacore
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 1.x
python_version: python_version
app_file: app.py
pinned: true
---

# 🌿 Permacore 🌿 Chatbot for Permaculture, Regenerative Agriculture, and Sustainable Developent

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

## Installation
A Python `venv` or `conda` environment is recommended for installation. Then, 

```bash
pip install -r requirements.txt
```

## Interactive UI
**COMING SOON - Hosting on HuggingFace**

The UI is powered by Gradio. A local UI can be started with by running `webUI.py`.

## Command Line Usage
**NOTICE - A MistralAI API key is required!**

```python
# Example usage
from Permacore.rag_agent import RAGAgent

agent = rag_agent()

# For a single query:
response = agent.query("What are the three ethics of permaculture?")
print(response)

# For interactive chat:
agent.chat()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

Suggestions for resources to share are also welcome!

## Data Sources

- [Open Source Ecology](https://wiki.opensourceecology.org/wiki/)
- [One Commuity Global](http://onecommunityglobal.org/)

## License

GPL-3.0
