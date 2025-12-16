import gradio as gr
from pathlib import Path

from rag_agent import RAGAgent, ModelConfig, RetrievalConfig


# Set custom theme
theme = gr.themes.Ocean(text_size="lg")

# Store conversation threads per session
conversation_threads = {}


def chat_with_agent(message, history, state, thread_id="default"):
    """
    Main chat interface function for Gradio ChatInterface.
    
    Args:
        message (str): User's input message
        history: Chat history
        thread_id (str): Conversation thread identifier
        
    Yields:
        str: Incremental chunks of the agent's response
    """

    response = ""

    for chunk in agent.stream_answer(message, history, state, thread_id=thread_id):
        response += chunk
        yield chunk

    return response, state


def query_agent(message, history, thread_id="default"):
    """
    Process user message and return response with sources.
    
    Args:
        message (str): User's input message
        history: Chat history (managed by Gradio)
        thread_id (str): Conversation thread identifier
        
    Returns:
        tuple: (response_text, sources_text)
    """
    if not message.strip():
        return "", "No sources yet."
    
    try:
        # Query the agent
        result = agent.query(message, thread_id=thread_id)
        
        # Format sources for display using shared helpers
        sources_text = "**Sources:**\n\n"
        for source in result["context"]:
            sources_text += f"{source}\n\n"
        
        return result["answer"], sources_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "Error retrieving sources."


def create_demo():
    """Create and configure the Gradio interface."""
    
    # Create the interface with tabs
    with gr.Blocks(theme=theme, title="RAG Agent Demo") as demo:
        
        gr.Markdown("""
        # 🌿 **Permacore** 🌿 Sustainable Development Knowledge Base
        
        Ask questions about sustainable development and get answers with source citations!
        
        Use the Chat tab for interactive conversations, or the Query with Sources tab to see detailed references.
        """)
        
        with gr.Tab("Chat"):
            state = gr.State({"context": None})

            chatbot = gr.ChatInterface(
                type="messages",
                fn=chat_with_agent,
                additional_inputs=[state],
                chatbot=gr.Chatbot(),
                title="Chat with Permacore",
                description="Ask questions about permaculture. " \
                "Type ```sources``` at any time to see the sources used in the last answer.",
                examples=[
                    ["What is permaculture?"],
                    ["What are the principles of permaculture?"],
                    ["How does permaculture relate to sustainability?"],
                    ["What are common permaculture techniques?"],
                ],
                # retry_btn=None,
                # undo_btn="◀️ Undo",
                # clear_btn="🗑️ Clear",
            )
        
        with gr.Tab("Query with Sources"):
            gr.Markdown("### Ask a question and see the sources")
            
            with gr.Row():
                with gr.Column(scale=2):

                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is permaculture?",
                        lines=3
                    )
                    query_btn = gr.Button("Ask Question", variant="primary")
                        # Example questions

                    gr.Examples(
                    examples=[
                        ["What is permaculture?"],
                        ["What are the ethics of permaculture?"],
                        ["Explain companion planting"],
                        ["What is forest gardening?"],
                    ],
                    inputs=query_input,
                )
                
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="Response will appear here after asking a question.",
                        show_copy_button=True
                    )
                with gr.Column():
                    sources_output = gr.Markdown(
                        label="📚 **RAG Sources**",
                        value="Sources will appear here after asking a question."
                    )
            
            
            # Connect the query button
            query_btn.click(
                fn=lambda q: query_agent(q, None, "query_tab"),
                inputs=query_input,
                outputs=[answer_output, sources_output]
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About Permacore
            
            This is a Retrieval-Augmented Generation (RAG) agent that answers questions about permaculture.
            
            ### Features:
            - 💬 **Interactive Chat**: Natural conversation with memory
            - 📚 **Source Citations**: Every answer includes references to source documents
            - 🔍 **Context-Aware**: Uses semantic search to find relevant information
            - 🧠 **Powered by**: Mistral AI and LangChain
            
            ### How it works:
            1. You ask a question
            2. The agent searches the knowledge base for relevant documents
            3. It generates an answer based on the retrieved context
            4. Sources are cited using [N] format
            
            ### Technology Stack:
            - **LLM**: Mistral AI
            - **Vector Store**: ChromaDB
            - **Framework**: LangChain
            - **UI**: Gradio

            ### Developed by:           
            Matt McCann :copyright: 2025 | GPL v3
                        
            View the [Source Code](https://github.com/mk-mccann/Permacore) on GitHub
                        
            [Personal Site](www.mk-mccann.github.io) | [LinkedIn](https://linkedin.com/in/matt-k-mccann/)
            """)
            
    
    return demo


if __name__ == "__main__":
    import argparse
    from os import getenv
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Run the RAG Agent Web UI")
    parser.add_argument(
        "--search_function",
        type=str,
        choices=["mmr", "similarity"],
        default="similarity",
        required=False,
        help="Search function to use for retrieval [options: mmr, similarity (default)]"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        required=False,
        help="Host address to run the Gradio app on."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        required=False,
        help="Port to run the Gradio app on."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate publicly sharable link."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode."
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    mistral_api_key = getenv("MISTRAL_API_KEY")

    if mistral_api_key:
        mistral_api_key = mistral_api_key.strip()
    else:   
        raise ValueError("MISTRAL_API_KEY not set in environment variables.")

    # Initialize the RAG agent
    agent = RAGAgent(
        chroma_db_dir=Path("../chroma_db"),
        collection_name="permacore",
        model_config=ModelConfig(),
        retrieval_config=RetrievalConfig(
                search_function=args.search_function, 
                similarity_threshold=0.3,  
            )
        )

    app = create_demo()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=args.debug,
        share=args.share,
    )
