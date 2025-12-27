from typing import Any, List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState

from config import RetrievalConfig
from utils.citation_formatter import build_citation, format_citation_line


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict
    context: list[tuple[Document, float | None]]


class RetrieveDocumentsMiddleware(AgentMiddleware[CustomAgentState]):
    state_schema = CustomAgentState

    def __init__(self, 
                 vectorstore: Chroma, 
                 retrieval_config: RetrievalConfig | None = None,
                 ):
        
        """
        Middleware to retrieve relevant documents before model invocation.

        Args:
            vectorstore (Chroma): Vector store for document retrieval.
            k_documents (int): Number of documents to retrieve.
            search_function (str): Search function to use 
                                   'mmr' for max marginal relevance
                                   (default) 'similarity' for standard similarity search.
            similarity_threshold (float): Similarity score threshold for document filtering.

        """
        
        # Initialize retrieval config with defaults if not provided
        if retrieval_config is None:
            retrieval_config = RetrievalConfig()

        self.vectorstore = vectorstore
        self.k_documents = retrieval_config.k_documents
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.lambda_mmr = retrieval_config.lambda_mmr

        if retrieval_config.search_function == "mmr":
            self.search_function = self._mmr_search
        elif retrieval_config.search_function == "similarity":
            self.search_function = self._similarity_search
        else:
            raise ValueError(f"Invalid search_function: {retrieval_config.search_function}. "
                             "Choose 'mmr' or 'similarity'.")


    def _similarity_search(self, query: str) -> list[tuple[Document, float]]:
        """
        Perform standard cosine distance search with score filtering. Lower is more similar.

        Args:
            query (str): Query string.
        Returns:
            list[tuple[Document, float]]: List of (Document, score) tuples.
        """

        retrieved_docs = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k_documents
        )

        if not retrieved_docs:
            return []
        
        # Filter docs based on similarity threshold
        passing_docs = [(doc, score) for (doc, score) in retrieved_docs if 0. < score < self.similarity_threshold]
        
        # Update doc metadata to flag that this comes from our database
        if not passing_docs:
            return []
        
        # Check if the source_type is already set to avoid overwriting
        if "source_type" in passing_docs[0][0].metadata:
            final_docs = passing_docs
        else:
            final_docs = [(doc.model_copy(update={"metadata": {**doc.metadata, "source_type": "rag"}}), score) for 
                        (doc, score) in passing_docs]

        return final_docs


    def _mmr_search(self, query: str) -> list[tuple[Document, None]]:
        """
        Perform maximum marginal relevance search. 
        Returns documents without scores since they aren't provided by MMR.
        
        Args:
            query (str): Query string.
        Returns:
            list[Tuple[Document, None]]: List of retrieved Documents.
        """

        retrieved_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k_documents,
            lambda_mult=self.lambda_mmr
        )
        
        # Check if the source_type is already set to avoid overwriting
        if "source_type" in retrieved_docs[0].metadata:
            docs = retrieved_docs
        else:
            docs = [doc.model_copy(update={"metadata": {**doc.metadata, "source_type": "rag"}}) for 
                    doc in retrieved_docs]

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

    def _format_retrieved_docs(self, 
                               retrieved_docs: List[Tuple[Document, float]] | List[Tuple[Document, None]],
                               original_query: Any) -> str:
        """
        Format retrieved documents into context string for model.
        
        Args:
            retrieved_docs: List of (Document, score) tuples from retrieval.
            original_query: Original user query content.
            
        Returns:
          str: Formatted context string for augmenting the model input.
        """

        # When feeding documents to the model, include citations
        # but not scores (not used in model input)
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


    # This overrides the before_model hook to inject retrieved documents
    def before_model(self, state: CustomAgentState, runtime: Any) -> dict[str, Any]:

        last_message = state["messages"][-1]

        # Extract query text
        query_text = self._extract_query_text(last_message.content)
        
        # Retrieve documents
        retrieved_docs = self.search_function(query_text)

        # Format documents locally (no RAGAgent dependency)
        augmented_content = self._format_retrieved_docs(
            retrieved_docs,
            last_message.content,
        )
        
        # Provide retrieved docs with scores under "context" (matches state_schema)
        return {
            "messages": [last_message.model_copy(update={"content": augmented_content})],
            "context": retrieved_docs,
        }
