import re
from typing import Any, List, Tuple, Dict, Literal, Union, Generator

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  

from utils.citation_formatter import format_context_for_display
from config import ChromaConfig, LLMConfig, RetrievalConfig
from middleware.retriever import CustomAgentState, RetrieveDocumentsMiddleware



class RAGAgent:

    def __init__(
        self,
        chroma_config: ChromaConfig = ChromaConfig(),
        llm_config: LLMConfig = LLMConfig(),
        retrieval_config: RetrievalConfig = RetrievalConfig(),
        **kwargs
    ):
        
        """
        Initialize the RAG agent with conversational memory.
        
        Args:
            chroma_config (ChromaConfig): Configuration for ChromaDB. If None, uses defaults.
            model_config (ModelConfig | None): Configuration for the LLM model. If None, uses defaults.
            retrieval_config (RetrievalConfig | None): Configuration for retrieval. If None, uses defaults.

        Kwargs:
            debug_score (bool): Flag to include similarity scores in the output for debugging. Default False.
        """
        
        self.chroma_config = chroma_config
        self.model_config = llm_config
        self.retrieval_config = retrieval_config
        
        # Debug flags
        self.debug_score = kwargs.get("debug_score", False)
        
        # Initialize embeddings
        self.embeddings = MistralAIEmbeddings(model=self.chroma_config.embeddings)
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name=self.chroma_config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_config.directory)
        )
        
        # Initialize chat model
        self.model = ChatMistralAI(
            model_name=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )
        
        # Now create the middleware with reference to self
        retriever = RetrieveDocumentsMiddleware(
                self.vectorstore,
                self.retrieval_config
            )

        # Initialize the agent
        self.agent = create_agent(
            self.model,
            system_prompt="Please be concise and to the point.",
            tools=[],
            middleware=[retriever],
            state_schema=CustomAgentState,
            checkpointer=InMemorySaver()
        )

        self.agent.invoke(
            {"messages": [{"role": "user", 
                           "content": "Hi! My source is Bob."}]},
            {"configurable": {"thread_id": "1"}},  
        )
        
        # Initialize context storage for sources command
        self._last_context: list[tuple[Document, float | None]] = []



    @staticmethod
    def _is_source_request(question: str) -> bool:
        """
        Check if the user is asking for sources.
        """
        source_keywords = [
            "source", "sources", "reference", "references",
            "where did you get this", "where is this from",
            "can you show me the sources", "what are your sources",
            "list the sources", "cite your sources", "citation", "citations",
            "bibliography", "footnotes", "what are the references"
        ]
        question_lower = question.lower()

        return any(keyword in question_lower for keyword in source_keywords)

    @staticmethod
    def _extract_llm_references(text: str) -> list:
        """Extract references like [1], [2], etc., from the LLM's response."""
        return re.findall(r'\[(\d+)\]', text)


    def query(self, question: str, thread_id: str = "default", **kwargs) -> dict:
        """
        Query the agent with a single question.
        
        Args:
            question (str): The question to ask.
            thread_id (str): Thread ID for conversation continuity.
            kwargs: Additional arguments (debug_score, include_content).
            
        Returns:
            dict: Contains 'answer' and 'context' keys.
        """
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}}
        )
        
        # Extract answer and context
        answer = result["messages"][-1].content
        retrieved = result.get("context", []) 

        if retrieved:
            context = format_context_for_display(retrieved, debug_score=self.debug_score)
        else:
            context = ["No sources retrieved."]

            
        return {
            "answer": answer,
            "context": context
        }


    def cli_chat(self, thread_id: str = "default"):
        """
        Start an interactive chat session. Used for CLI interface.
        
        Args:
            thread_id (str): Unique identifier for this conversation thread.
        """

        print("RAG Agent Chat Interface")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("-" * 50)

        last_context = None    # Initialize last_context to None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'sources':
                    if last_context:
                        print("\nSources from last response:")
                        for citation in format_context_for_display(last_context, 
                                                                   debug_score=self.debug_score):
                            print(citation)
                    else:
                        print("No sources available yet. Ask a question first!")
                    continue
                
                if not user_input:
                    continue
                
                # Stream the response
                print("\nAssistant: ", end="", flush=True)
                
                for step in self.agent.stream(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                    },
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="values"
                ):
                    # Get the last message
                    last_msg = step["messages"][-1]
                    
                    # Only print assistant messages
                    if last_msg.type == "ai":
                        # Print content (this will show incrementally if streaming)
                        print(last_msg.content, end="", flush=True)
                        
                        # Save context for 'sources' command
                        if "context" in step:
                            last_context = step["context"]
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


    def stream_answer(self, 
                      question: str, 
                      history: Union[List[Tuple[str, str]], List[Dict[Literal["role", "content"], str]]],
                      state: Dict[str, Any],
                      thread_id: str = "default"
                      ) -> Generator[str, None, None]:

        """
        Stream a single answer (with optional sources at the end) for UI use.

        Yields incremental text chunks suitable for Gradio's streaming handlers.
        """

        partial = ""
        full_response = ""
        rag_context = []

        # Handle "sources" command
        if self._is_source_request(question):
            if state.get("context"):
                sources = format_context_for_display(state["context"])
                yield "\n\nSources:\n" + "\n".join(sources)
            else:
                yield "\n\nNo RAG sources available."
            return

        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="values"
            ):

            msg = step["messages"][-1]

            # Accumulate incremental assistant text
            if msg.type == "ai":
                if msg.content != partial:  # Only yield if content has changed
                    if partial:  # If we have previous content, yield the difference
                        new_text = msg.content[len(partial):]
                        if new_text:
                            yield new_text
                            full_response += new_text

                    else:  # First chunk, yield the whole content
                        yield msg.content
                        full_response += msg.content
                    partial = msg.content

            # Capture context for sources when available
            if "context" in step:
                # Store context for potential "sources" command
                rag_context = [(doc, score) for doc, score in step["context"] if doc.metadata.get("source_type") == "rag"]
                state["context"] = rag_context

        # # After streaming, check for LLM-generated references
        # if full_response: 
        #     references = self._extract_llm_references(full_response)
        #     if references:
        #         rag_source_ids = {str(i+1) for i, _ in enumerate(state["context"])}
        #         for ref in references:
        #             if ref not in rag_source_ids:
        #                 full_response += f"\n\n⚠️ Reference [{ref}] is not from the RAG database."

    
    def _test_query_prompt_with_context(self):
        query = (
            "What is permaculture?"
        )

        # Create properly typed state
        input_state: CustomAgentState = {
            "messages": [{"role": "user", "content": query}],
            "user_id": "user_123",
            "preferences": {"theme": "dark"},
            "context": []
        }

        for step in self.agent.stream(
            input_state,
            {"configurable": {"thread_id": "1"}},
            stream_mode="values"
            ):

            step["messages"][-1].pretty_print()


    def _test_query_with_sources(self):
        """Test query that shows citations."""
        query = "What are Holmgren's 12 Design Principles?"
        
        print("Question:", query)
        print("\n" + "="*50 + "\n")
        
        result = self.query(query, thread_id="test")
        
        print("Answer:")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")
        print("Sources:")

        for source in result["context"]:
            print(source)


if __name__ == "__main__":
    from os import getenv
    from dotenv import load_dotenv
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG Agent in the terminal or test it with a query.")

    parser.add_argument("--chat", action="store_true", help="Run the RAG Agent in interactive chat mode.")
    parser.add_argument("--test", action="store_true", help="Test the RAG Agent with a predefined query.")
    parser.add_argument("--debug_scores", action="store_true", default=False, help="Display similarity scores in sources (for testing).")

    args = parser.parse_args()

    load_dotenv()
    mistral_api_key = getenv("MISTRAL_API_KEY")

    if mistral_api_key:
        mistral_api_key = mistral_api_key.strip()
    else:   
        raise ValueError("MISTRAL_API_KEY not set in environment variables.")
    

    # Set up configurations
    chroma_config = ChromaConfig()
    
    llm_config = LLMConfig(
        model="mistral-small-latest",
        temperature=0.7,
        max_tokens=1024
        )
    
    retrieval_config = RetrievalConfig(
        debug_score=args.debug_scores
        )

    # Initialize the RAG agent
    agent = RAGAgent(
        chroma_config=chroma_config,
        llm_config=llm_config,
        retrieval_config=retrieval_config
    )
    
    # Option 1: Interactive chat
    if args.chat:
        agent.cli_chat(thread_id="session_1")
    
    # Option 2: Single query with sources
    elif args.test:
        agent._test_query_with_sources()

    else:
        parser.print_help()
