import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# ====================================
# Configuration constants
# ====================================

DISALLOWED_FILE_TYPES = [
    '.png', '.jpg', '.jpeg', '.gif', '.pdf', 
    '.docx', '.zip', '.fcstd', '.stl', '.kdenlive',
    '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp',
    '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
    '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
    '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds',
    '.ino', '.stp'
]

DISALLOWED_PAGE_TYPES = [
    'File:', 'Schematic:', 'Category:', 'Special:', 
    'Template:', 'one-community-welcomes'
]

BOILERPLATE_INDICATORS = [
    "Navigation menu",
    "Contribute to this page",
    "###### WHO WE ARE",
    "###### WHO IS ONE COMMUNITY",
    "Retrieved from",
]

FRONTMATTER_KEY_ORDER = [
    'source', 'title', 'author', 'url', 'access_date', 
    'date', 'license', 'description', 'keywords'
]


# ====================================
# Data classes used throughout this project
# ====================================

CONFIG_PATH = Path("configs/config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config


@dataclass
class ChromaConfig:
    """
    Configuration for ChromaDB

    Args:
        collection_name (str): Name of ChromaDB collection.
        directory (Path | str): Directory to store the ChromaDB database.
        metadata (Dict[str, str]): Metadata for the collection, e.g. embedding function: {"hnsw:space": "cosine"}.
    """

    embeddings = None  
    directory: str | Path = Path("../chroma")
    collection_name: str = "default_collection"
    metadata: Dict[str, str] = field(default_factory = lambda: {"hnsw:space": "cosine"})

    # def __init__(self, **kwargs):
    #     # Load user config from JSON
    #     user_config = self._load_config()
    #     user_config = user_config.get('chroma', {})
        
    #     # Override default values with user config if provided
    #     for key, value in {**user_config, **kwargs}.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)

    #     # Ensure directory exists
    #     if type(self.directory) is str:
    #         self.directory = Path(self.directory)

    #     self.directory.mkdir(parents=True, exist_ok=True)    #type: ignore

    # @staticmethod
    # def _load_config():
    #     return load_config()



@dataclass
class LLMConfig:
    """
    Configuration for the LLM model.

    Args:
        name (str): Model name. Default is "mistral-small-latest".
        temperature (float): Model temperature. Default is 0.7.
        max_tokens (int): Maximum number of tokens for the model output. Default is 1024.
    """

    model: str = "mistral-small-latest"
    temperature: float = 0.7
    max_tokens: int = 1024

    # def __init__(self, **kwargs):
    #     # Load user config from JSON
    #     user_config = self._load_config()
    #     user_config = user_config.get('llm', {})

    #     # Override default values with user config if provided
    #     for key, value in {**user_config, **kwargs}.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)

    # @staticmethod
    # def _load_config():
    #     return load_config()


@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval.
    
    Args:
        k_documents (int): Number of documents to retrieve.
        search_function (str): Search function to use
                                      'mmr' for max marginal relevance
                                      (default) 'similarity' for standard similarity search.    
        similarity_threshold (float): Similarity score threshold for document filtering.
        lambda_mmr (float): Lambda parameter for max marginal relevance search.
        debug_score (bool): Flag to include similarity scores in the output for debugging.
    """

    k_documents: int = 5
    search_function: str = "similarity"   # 'mmr' for max marginal relevance. Default is cosine similarity search
    similarity_threshold: float = 0.5     # used for similarity search
    lambda_mmr: float = 0.7  # used for mmr search
    debug_score: bool = False

    # def __init__(self, **kwargs):
    #     # Load user config from JSON
    #     user_config = self._load_config()
    #     user_config = user_config.get('retrieval', {})

    #     # Override default values with user config if provided
    #     for key, value in {**user_config, **kwargs}.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)

    # @staticmethod
    # def _load_config():
    #     return load_config()
