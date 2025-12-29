import json
import yaml
from pathlib import Path
from os import getenv
from dotenv import load_dotenv
from typing import Dict, Any, TypeVar, Type
from pydantic import BaseModel, Field


# Set default project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT_DIR / "chroma"
DATA_DIR = ROOT_DIR / "data"
HF_DATA_DIR = ROOT_DIR / "hf_data"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_PATH = ROOT_DIR / "configs" / "user_config.yaml"

# Load environment variables from .env file if it exists
load_dotenv()

if getenv("MISTRAL_API_KEY"):
    MISTRAL_API_KEY = getenv("MISTRAL_API_KEY").strip()
else:   
    raise ValueError("MISTRAL_API_KEY not set in environment variables.")

# Gradio server configuration from environment variables. Fall back to defaults if not set.
GRADIO_SERVER_PORT = int(getenv("GRADIO_SERVER_PORT")) if getenv("GRADIO_SERVER_PORT") else 7860
GRADIO_SERVER_NAME = getenv("GRADIO_SERVER_NAME") if getenv("GRADIO_SERVER_NAME") else "127.0.0.1"

# ============================================
# Configuration constants
# ============================================

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


# ============================================
# Data classes used throughout this project
# Defualts can be overridden by user config
# ============================================

T = TypeVar('T', bound='ConfigBase')

class ConfigBase(BaseModel):
    """
    Base class for all configurable Pydantic models.
    Provides a method to load configuration from a file and override with CLI arguments.
    """

    @classmethod
    def from_config(cls: Type[T], config_path: str | Path, key: str, **cli_overrides) -> T:
        """Load configuration from a YAML or JSON file and override defaults."""

        config_path = Path(config_path)

        if not config_path.exists():
            print(f"Config file {config_path} does not exist. Using default configuration.")
            return cls()  # Return defaults if file doesn't exist

        with open(config_path, "r") as f:
            if config_path.suffix in (".yaml", ".yml"):
                user_config = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                user_config = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use YAML or JSON.")
            
        base_config = cls(**user_config[key])

        # Override with CLI arguments
        return base_config.model_copy(update=cli_overrides)


class ChromaConfig(ConfigBase):
    """
    Configuration for ChromaDB

    Args:
        embeddings (Any): Embedding model or function to use. User can provide their own embedding function.
        directory (Path | str): Directory to store the ChromaDB database.
        collection_name (str): Name of ChromaDB collection.
        metadata (Dict[str, str]): Metadata for the collection, e.g. embedding function: {"hnsw:space": "cosine"}.
    """

    embeddings: Any = None
    directory: str | Path = CHROMA_DIR
    collection_name: str = "default_collection"
    metadata: Dict[str, str] = Field(default_factory = lambda: {"hnsw:space": "cosine"})


class LLMConfig(ConfigBase):
    """
    Configuration for the LLM model. This example uses MistralAI.

    Args:
        model_name (str): Model name. Default is "mistral-small-latest".
        temperature (float): Model temperature. Default is 0.7.
        max_tokens (int): Maximum number of tokens for the model output. Default is 1024.
    """

    model_name: str = "mistral-small-latest"
    temperature: float = 0.7
    max_tokens: int = 1024


class RetrievalConfig(ConfigBase):
    """
    Configuration for document retrieval.
    
    Args:
        k_documents (int): Number of documents to retrieve.
        search_function (str): Search function to use
                                      'mmr' for max marginal relevance
                                      (default) 'similarity' for standard cosine similarity search.    
        similarity_threshold (float): Similarity score threshold for document filtering.
        lambda_mmr (float): Lambda parameter for max marginal relevance search.
        debug_score (bool): Flag to include similarity scores in the output for debugging.
    """

    k_documents: int = 5
    search_function: str = "similarity"   # options: 'mmr', 'similarity'
    similarity_threshold: float = 0.3     # used for similarity search
    lambda_mmr: float = 0.7               # used for mmr search


if __name__ == "__main__":
    # Example usage
    chroma_config = ChromaConfig.from_config(CONFIG_PATH, "chroma")
    llm_config = LLMConfig.from_config(CONFIG_PATH, "llm")
    retrieval_config = RetrievalConfig.from_config(CONFIG_PATH, "retrieval")

    print("Loading configurations from", CONFIG_PATH)
    print("Chroma Config:", chroma_config)
    print("LLM Config:", llm_config)
    print("Retrieval Config:", retrieval_config)
