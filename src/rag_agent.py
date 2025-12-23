import re
from pathlib import Path
from typing import Any, List, Tuple, Dict, Literal, Union, Generator

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.checkpoint.memory import InMemorySaver  

from utils.citation_formatter import build_citation, format_citation_line, format_context_for_display
from src.config import LLMConfig, RetrievalConfig



class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict
    context: list[tuple[Document, float | None]]


class RetrieveDocumentsMiddleware(AgentMiddleware[CustomAgentState]):

    state_schema = CustomAgentState

    def __init__(self, 
                 rag_agent: 'RAGAgent',
                 vectorstore: Chroma, 
                 k_documents: int = 4, 
                 search_function: str = "similarity",
                 similarity_threshold: float = 0.25,    # used for similarity search
                 lambda_mmr: float = 0.5    # used for mmr search
                 ):
        
        """
        Middleware to retrieve relevant documents before model invocation.

        Args:
            rag_agent (RAGAgent): Reference to parent RAG agent for formatting.
            vectorstore (Chroma): Vector store for document retrieval.
            k_documents (int): Number of documents to retrieve.
            search_function (str): Search function to use 
                                   'mmr' for max marginal relevance
                                   (default) 'similarity' for standard similarity search.
            similarity_threshold (float): Similarity score threshold for document filtering.

        """
        
        self.vectorstore = vectorstore
        self.rag_agent = rag_agent
        self.k_documents = k_documents
        self.similarity_threshold = similarity_threshold
        self.lambda_mmr = lambda_mmr

        if search_function == "mmr":
            self.search_function = self._mmr_search
        else:
            self.search_function = self._similarity_search


    def _similarity_search(self, query: str) -> list[tuple[Document, float]]:
        """
        Perform standard similarity search with score filtering.

        Args:
            query (str): Query string.
        Returns:
            list[tuple[Document, float]]: List of (Document, score) tuples.
        """

        retrieved_docs_scores =  self.vectorstore.similarity_search_with_score(
            query,
            k=self.k_documents
        )

        # Update doc metadata to flag that this comes from our database
        docs = [(doc.model_copy(update={"metadata": {**doc.metadata, "source_type": "rag"}}), score) for 
                (doc, score) in retrieved_docs_scores]

        return [(doc, score) for (doc, score) in docs 
                           if 0 < score < self.similarity_threshold]


    def _mmr_search(self, query: str) -> list[tuple[Document, None]]:
        """
        Perform maximum marginal relevance search. 
        Returns documents without scores since they aren't provided by MMR.
        
        Args:
            query (str): Query string.
        Returns:
            list[Tuple[Document, None]]: List of retrieved Documents.
        """

        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k_documents,
            lambda_mult=self.lambda_mmr
        )
        
        # Update doc metadata to flag that this comes from our database
        docs = [doc.model_copy(update={"metadata": {**doc.metadata, "source_type": "rag"}}) for doc in docs]

        # Now return documents with None as score placeholder for consistency
        return [(doc, None) for doc in docs]


    def _extract_query_text(self, content: Any) -> str:
        """Extract query text from message content of various types."""
        if isinstance(content, str):
            return content
        
        elif isinstance(content, list):
            text_parts = [
                item if isinstance(item, str) else item.get("text", "")
                for item in content
            ]
            return " ".join(text_parts).strip()
        else:
            return str(content)


    # This overrides the before_model hook to inject retrieved documents
    def before_model(self, state: CustomAgentState, runtime: Any) -> dict[str, Any]:

        last_message = state["messages"][-1]

        # Extract query text
        query_text = self._extract_query_text(last_message.content)
        
        # Retrieve documents
        retrieved_docs = self.search_function(query_text)

        # Use RAG agent's formatting method with debug_score flag
        augmented_content = self.rag_agent._format_retrieved_docs(
            retrieved_docs,
            last_message.content,
        )
        
        # Provide retrieved docs with scores under "context" (matches state_schema)
        return {
            "messages": [last_message.model_copy(update={"content": augmented_content})],
            "context": retrieved_docs,
        }


# class TrimMessagesMiddleware(AgentMiddleware[CustomAgentState]):

#     state_schema = CustomAgentState

#     @before_model
#     def trim_messages(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
#         """Keep only the last few messages to fit context window."""
#         messages = state["messages"]

#         if len(messages) <= 3:
#             return None  # No changes needed

#         first_msg = messages[0]
#         recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
#         new_messages = [first_msg] + recent_messages

#         return {
#             "messages": [
#                 RemoveMessage(id=REMOVE_ALL_MESSAGES),
#                 *new_messages
#             ]
#         }


#     def delete_specfic_messages(state):
#         messages = state["messages"]
#         if len(messages) > 2:
#             # remove the earliest two messages
#             return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  
        

#     def delete_all_messages(state):
#         return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}  


class RAGAgent:

    def __init__(
        self,
        chroma_db_dir: Path | str,
        collection_name: str = "ragrarian",
        embeddings_model: str = "mistral-embed",
        llm_config: LLMConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
        **kwargs
    ):
        
        """
        Initialize the RAG agent with conversational memory.
        
        Args:
            chroma_db_dir (Path | str): Directory containing the ChromaDB database.
            collection_name (str): Name of the ChromaDB collection. Default is 'ragrarian'.
            embeddings_model (str): Model to use for embeddings. Default is 'mistral-embed'.
            model_config (ModelConfig | None): Configuration for the LLM model. If None, uses defaults.
            retrieval_config (RetrievalConfig | None): Configuration for retrieval. If None, uses defaults.

        Kwargs:
            debug_score (bool): Flag to include similarity scores in the output for debugging. Default False.
        """
        
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = collection_name
        self.model_config = llm_config or LLMConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        
        # Debug flags
        self.debug_score = kwargs.get("debug_score", False)
        
        # Initialize embeddings
        self.embeddings = MistralAIEmbeddings(model=embeddings_model)
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_db_dir)
        )
        
        # Initialize chat model
        self.model = ChatMistralAI(
            model_name=self.model_config.model,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )

        # Create agent with retrieval middleware
        # Note: self.agent is initialized after middleware to avoid circular reference issues
        self.agent = None
        
        # Now create the middleware with reference to self
        retriever = RetrieveDocumentsMiddleware(
                self,
                self.vectorstore,
                self.retrieval_config.k_documents,
                self.retrieval_config.search_function,
                self.retrieval_config.similarity_threshold,
                self.retrieval_config.lambda_mmr
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


    def _format_retrieved_docs(self, 
                               retrieved_docs: List[Tuple[Document, float]] | List[Tuple[Document, None]],
                               original_query: Any, 
                               ) -> str:
        """
        Format retrieved documents into context string for model.
        
        Args:
            retrieved_docs: List of (Document, score) tuples from retrieval.
            original_query: Original user query content.
            
        Returns:
            tuple: (formatted_context_string, list_of_docs_for_state)
        """

        # When feeding the documents to the model, we want to include the citation
        # but not the score (if provided) as it's not used in the model input.
        docs_content_with_citations = []
        for idx, (doc, score) in enumerate(retrieved_docs, 1):
            citation = build_citation(doc, idx, score)
            docs_content_with_citations.append(
                format_citation_line(citation, include_content=doc.page_content)
            )
        
        docs_content = "\n\n".join(docs_content_with_citations)
        
        augmented_content = (
            f"{original_query}\n\n"
            "Use the following context to answer the query. "
            "When using information from the context, cite the source number (e.g., [1]):\n\n"
            f"{docs_content}"
        )
        
        return augmented_content
    
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

    agent = RAGAgent(
        chroma_db_dir=Path("../chroma_db"),
        collection_name="permacore",
        embeddings_model="mistral-embed",
        llm_config=LLMConfig(
            model="mistral-small-latest",
            temperature=0.7,
            max_tokens=1024
        ),
        retrieval_config=RetrievalConfig(
            k_documents=5,
            search_function="similarity", 
            similarity_threshold=10000,  # High threshold to allow all documents
            debug_score=args.debug_scores
        )
    )
    
    # Option 1: Interactive chat
    if args.chat:
        agent.cli_chat(thread_id="session_1")
    
    # Option 2: Single query with sources
    elif args.test:
        agent._test_query_with_sources()

    else:
        parser.print_help()
