import os
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader, PyMuPDFLoader


class DocumentIngestor:
    """
    Ingest documents from various formats into LangChain Document objects.
    Currently supports PDF files.
    """

    def __init__(self, 
                 data_dir: Path|str):
        
        self.data_dir = Path(data_dir)


    def ingest_pdf(self, filepath: Path|str) -> Document | List[Document]:
        """
        Ingest a PDF file and return a LangChain Document. Defaults to using 
        UnstructuredPDFLoader, with a fallback to PyMuPDFLoader if Unstructured fails.
        
        Args:
            filepath (Path|str): Path to the PDF file.

        Returns:
            Document | List[Document]: LangChain Document object containing the PDF content.
        """

        filepath = Path(filepath)

        try:
            # Try UnstructuredPDFLoader first
            loader = UnstructuredPDFLoader(filepath, mode="elements", strategy="hi_res")
        
        except Exception as e:
            # Fallback to PyMuPDFLoader if Unstructured fails
            print(f"Falling back to PyMuPDF: {e}")
            loader = PyMuPDFLoader(filepath)

        docs = loader.load()
        if not docs:
            raise ValueError(f"No content extracted from {filepath}")
        
        return docs
