from time import sleep
from pathlib import Path
import json
import hashlib
import random

from httpx import ReadError
from alive_progress import alive_it, alive_bar
from typing import List, Dict, Set, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_mistralai.embeddings import MistralAIEmbeddings


class CreateChromaDB:

    def __init__(
        self, 
        embeddings,
        chunked_docs_dir: Path | str, 
        chroma_db_dir: Path | str, 
        checkpoint_file: Path | str = "embedding_progress.json",
        failed_batches_file: Path | str = "failed_batches.txt",
        collection_name: str = "default_collection",
        index_file: Optional[Path | str] = None,
        ):
       
        """
        Create a ChromaDB vector database from chunked JSONL files in a specified directory.

        Args:
            embeddings: Embedding function to use for vectorization.
            chunked_docs_dir (Path | str): Directory containing markdown files.
            chroma_db_dir (Path | str): Directory to store the ChromaDB database.
            checkpoint_file (str): File to store processing progress.
            failed_batches_file (str): File to log failed batches.
            collection_source (str): source of the ChromaDB collection.
            index_file (Path | str): Path to the index.json file that tracks file hashes.
        """

        self.embeddings = embeddings
        self.collection_name = collection_name
        self.chunked_docs_dir = Path(chunked_docs_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.failed_batches_file = Path(failed_batches_file)
        
        # Set default index file path if not provided
        if index_file is None:
            self.index_file = self.chunked_docs_dir / "index.json"
        else:
            self.index_file = Path(index_file)

        if not self.chroma_db_dir.exists():
            self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name = self.collection_name,
            embedding_function = embeddings,
            persist_directory = str(self.chroma_db_dir)
        )    


    def refresh_embeddings(self):
        """
        Refresh embeddings in the ChromaDB vector database.
        """

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_db_dir)
        )


    def load_index(self) -> Dict:
        """
        Load the index.json file that tracks file hashes and metadata.
        
        Returns:
            Dictionary containing file metadata and hashes.
        """
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}


    def get_modified_and_new_files(self) -> Tuple[Set[str], Set[str]]:
        """
        Compare current files with the index to find new and modified files.
        
        Returns:
            Tuple of (new_files, modified_files) as sets of cache file paths.
        """
        index = self.load_index()
        current_files = set()
        new_files = set()
        modified_files = set()
        
        # Get all current JSONL files
        jsonl_files = list(self.chunked_docs_dir.glob("**/*.jsonl"))
        
        for file_path in jsonl_files:
            # Get relative path for matching with index
            rel_path = file_path.relative_to(self.chunked_docs_dir)
            cache_file = str(rel_path)
            current_files.add(cache_file)
            
            file_in_index = False
            for key, value in index.items():
                if value.get('cache_file') == cache_file:
                    file_in_index = True
                    mtime = value.get('mtime')
                    size = value.get('size')
                    content_hash = value.get('content_hash')
                    source_path = value.get('file_path')

                    # Check for modifications using mtime/size/content_hash
                    if source_path and (mtime is not None and size is not None and content_hash):
                        
                        try:
                        
                            # Fast check: mtime/size
                            st = Path(source_path).stat()
                        
                            if (st.st_mtime != mtime) or (st.st_size != size):
                        
                                # Confirm modification with content hash
                                h = hashlib.sha256()
                        
                                with open(source_path, 'rb') as f:
                                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                                        h.update(chunk)
                                current_hash = h.hexdigest()
                        
                                if current_hash != content_hash:
                                    modified_files.add(cache_file)
                        
                            # If mtime/size match, assume unchanged; skip hash
                        
                        except Exception:
                            # Missing source or IO issue: mark as modified to be safe
                            modified_files.add(cache_file)
                    
                    else:
                    
                        # Fallback: compare JSONL mtime if index lacks full hash data
                        try:
                            current_jsonl_mtime = file_path.stat().st_mtime
                            indexed_jsonl_mtime = value.get('jsonl_mtime')
                   
                            if indexed_jsonl_mtime is None or indexed_jsonl_mtime != current_jsonl_mtime:
                                modified_files.add(cache_file)
                   
                        except Exception:
                            modified_files.add(cache_file)
                   
                    break
            
            if not file_in_index:
                new_files.add(cache_file)
        
        return new_files, modified_files


    def load_chunked_documents(self, file_filter: Optional[Set[str]] = None):
        """
        Recursively load chunked JSONL documents from the specified directory.
        
        Args:
            file_filter: Set of relative file paths to load. If None, loads all files.

        Returns:
            List of loaded documents.
        """

        all_docs = []
        jsonl_files = list(self.chunked_docs_dir.glob("**/*.jsonl"))
        
        # Filter files if file_filter is provided
        if file_filter is not None:
            filtered_files = []
            for file_path in jsonl_files:
                rel_path = str(file_path.name)
                if rel_path in file_filter:
                    filtered_files.append(file_path)
            jsonl_files = filtered_files

        for file_path in alive_it(jsonl_files, title="Loading chunked documents for ChromaDB"):
            loader = JSONLoader(
                file_path=str(file_path),     
                jq_schema='.',
                content_key='page_content',
                json_lines=True,
                metadata_func=lambda record, metadata: record.get('metadata', {}))
                
            docs = loader.load()
            docs = filter_complex_metadata(docs)
            all_docs.extend(docs)

        # Remove any leading/trailing whitespace from document contents and empty entries
        all_docs = [doc for doc in all_docs if doc.page_content.strip()]

        return all_docs
    

    def load_checkpoint(self):
        """
        Load the last checkpoint to resume processing.
        
        Returns:
            int: The index to start processing from.
        """
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint['last_processed']
                print(f"Resuming from index {start_idx}")
                return start_idx
        except FileNotFoundError:
            print("No checkpoint found. Starting from beginning.")
            return 0


    def save_checkpoint(self, index: int, total_docs: int, batch_size: int):
        """
        Save the current processing checkpoint.
        
        Args:
            index (int): Current processing index.
            total_docs (int): Total number of documents.
            batch_size (int): Batch size being used.
        """
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'last_processed': index,
                'total_docs': total_docs,
                'batch_size': batch_size
            }, f)


    def log_failed_batch(self, batch_num: int, documents: List[Document], error: Exception):
        """
        Log details of a failed batch including JSONL file paths for reliable retry.
        
        Args:
            documents (List[Document]): Documents in the batch.
            error (Exception): The error that occurred.
        """
        with open(self.failed_batches_file, 'a') as f:
            f.write(f"Batch {batch_num}\n")
            
            # Store JSONL cache file paths (relative to chunked_docs_dir)
            jsonl_files = set()
            for doc in documents:
                if 'file_path' in doc.metadata:
                    # Map original markdown file to its JSONL cache equivalent
                    # The cache is indexed by parent_dir/file_stem.jsonl
                    orig_path = Path(doc.metadata['file_path'])
                    parent_dir = orig_path.parent.name
                    file_stem = orig_path.stem
                    cache_file = f"{parent_dir}/{file_stem}.jsonl"
                    jsonl_files.add(cache_file)
            if jsonl_files:
                f.write(f"Files: {','.join(sorted(jsonl_files))}\n")
            f.write(f"Error: {str(error)}\n")
            f.write("-" * 50 + "\n")


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ReadError,))
    )
    def add_batch_with_retry(self, batch: List[Document]):
        """
        Add a batch of documents with retry logic.
        
        Args:
            batch: List of documents to add.
        """
        self.vectorstore.add_documents(batch)


    def remove_documents_by_source(self, source_files: Set[str]):
        """
        Remove documents from the vector store based on source file paths.
        
        Args:
            source_files: Set of source file paths to remove.
        """
        if not source_files:
            return
            
        print(f"Removing {len(source_files)} modified files from the database...")
        
        # Get all documents with their IDs
        collection = self.vectorstore._collection
        
        # Get all data from collection
        all_data = collection.get(include=['metadatas'])
        
        if not all_data['ids']:
            print("No existing documents found in collection.")
            return
        
        # Find IDs to delete based on source file
        ids_to_delete = []
        metadatas = all_data.get('metadatas', [])
        ids = all_data.get('ids', [])
        
        if not metadatas or not ids:
            print("No metadata found in collection.")
            return
            
        for idx, metadata in enumerate(metadatas):
            if metadata and 'file_path' in metadata:
                file_path = metadata.get('file_path')
                if file_path and isinstance(file_path, str):
                    # Check if this document's source file is in our set of files to remove
                    for source_file in source_files:
                        if source_file in file_path:
                            ids_to_delete.append(ids[idx])
                            break
        
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} document chunks from modified sources...")
            collection.delete(ids=ids_to_delete)
            print("Deletion complete.")
        else:
            print("No matching documents found to delete.")


    def embed_and_store(self, 
                        batch_size: int = 100, 
                        delay_seconds: float = 1,
                        resume: bool = True,
                        rebuild: bool = False):
        """
        Embed and store the loaded documents into the ChromaDB vector database in batches.
        
        Args:
            batch_size (int): Number of documents to process in each batch. Default is 100.
            delay_seconds (float): Seconds to wait between batches. Default is 1.
            resume (bool): Whether to resume from checkpoint. Default is True.
            rebuild (bool): Whether to rebuild the whole database or only process new/modified files. Default is False.
        """
        
        # Determine which files to process
        file_filter = None
        modified_source_files = set()
        
        if not rebuild:
            # Check that index file exists
            index = self.load_index()

            if not index:
                print("Index file missing or empty. Performing full rebuild.")
                rebuild = True
                del index

            else:
                print("Checking for new and modified files...")
                new_files, modified_files = self.get_modified_and_new_files()
                
                if not new_files and not modified_files:
                    print("No new or modified files found. Database is up to date.")
                    return
                
                print(f"Found {len(new_files)} new files and {len(modified_files)} modified files.")
                
                # Combine new and modified files
                file_filter = new_files.union(modified_files)
                
                # For modified files, we need to remove old embeddings first
                if modified_files:
                    # Extract source file paths from index
                    index = self.load_index()
                    for key, value in index.items():
                        cache_file = value.get('cache_file', '')
                        if cache_file in modified_files:
                            source_file = value.get('file_path', '')
                            if source_file:
                                modified_source_files.add(source_file)
                    
                    # Remove old embeddings for modified files
                    self.remove_documents_by_source(modified_source_files)
        
        # Load documents (filtered if incremental)
        documents = self.load_chunked_documents(file_filter=file_filter)
        
        if not documents:
            print("No documents to process.")
            return
            
        print(f"Loaded {len(documents)} documents to embed")
        
        # Determine starting index
        start_idx = self.load_checkpoint() if resume else 0
        
        # Calculate total batches for progress tracking
        batches_to_process = ((len(documents) - start_idx) + batch_size - 1) // batch_size
        
        # Initialize i for exception handling
        i = start_idx
        
        try:
            with alive_bar(batches_to_process, dual_line=True, title="Batches embedded") as bar:
                for i in range(start_idx, len(documents), batch_size):

                    batch = documents[i:i + batch_size]
                    batch_num = i // batch_size + 1
                                        
                    try:
                        self.add_batch_with_retry(batch)
                        
                        # Save checkpoint after successful batch
                        self.save_checkpoint(i + batch_size, len(documents), batch_size)
                        
                        # Update progress bar
                        bar()

                        # Rate limiting delay
                        sleep(delay_seconds)
                        
                        
                    except Exception as e:
                        print(f"\nBatch {batch_num} failed after retries: {e}")
                        
                        # Log failed batch (pass the batch documents for file path logging)
                        self.log_failed_batch(batch_num, batch, e)
                        
                        print(f"Logged to {self.failed_batches_file}. Continuing with next batch...")
                        
                        # Update checkpoint to skip this batch
                        self.save_checkpoint(i + batch_size, len(documents), batch_size)

                        # Update progress bar
                        bar()
                        
                        # Longer delay after failure
                        sleep(5)
                        continue

        except KeyboardInterrupt:
            print(f"\n\nProcess interrupted. Progress saved at index {i}.")
            print(f"Run again with resume=True to continue from this point.")
            raise

        print("\n✓ Embedding complete!")
        print(f"Processed {len(documents)} documents")
        
        # Check for failures
        try:
            with open(self.failed_batches_file, 'r') as f:
                failures = f.read()
                if failures:
                    print(f"\n⚠ Some batches failed. Check {self.failed_batches_file} for details.")
        except FileNotFoundError:
            print("No failures recorded.")


    def retry_failed_batches(self, batch_size: int = 100, delay_seconds: float = 1):
        """
        Read the failed batches log and retry embedding only those documents.
        Only loads JSONL files that contain failed documents.
        
        Args:
            batch_size (int): Number of documents per batch.
            delay_seconds (float): Seconds to wait between retries.
        """

        try:
            with open(self.failed_batches_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("No failed batches log found. Nothing to retry.")
            return

        # Extract unique JSONL file paths from failed batches
        failed_jsonl_files = set()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Files:"):
                files_str = line.split("Files: ")[1]
                # for a temporary fix, we're gonna manually change the md file paths to jsonl paths
                files = [f.strip() for f in files_str.split(",")]
                files = [f.replace(".md", ".jsonl") for f in files]
                files = [Path(f.replace("/raw/", "/chunked_documents/")) for f in files]
                files = [str(f.name) for f in files]
                failed_jsonl_files.update(files)
                # failed_jsonl_files.update(f.strip() for f in files_str.split(","))
            i += 1

        if not failed_jsonl_files:
            print("No JSONL file information in failed batches log. Cannot retry efficiently.")
            return

        print(f"Loading documents from {len(failed_jsonl_files)} failed JSONL files...")
        
        # Load only documents from failed JSONL files
        documents = self.load_chunked_documents(file_filter=failed_jsonl_files)

        # For funsies, shuffle them around
        random.shuffle(documents)
        total_batches = (len(documents) + batch_size - 1) // batch_size
    
        if not documents:
            print("No documents found to retry.")
            return

        print(f"Retrying {len(documents)} failed documents in {total_batches} batches...")

        failed_retries = 0
        with alive_bar(total_batches, dual_line=True) as bar:
            for i in range(0, len(documents), batch_size):

                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    self.add_batch_with_retry(batch)
                    sleep(delay_seconds)
                
                except Exception as e:
                    print(f"\nRetry failed for batch {batch_num}: {e}")
                    failed_retries += 1
                    
                    # Log the failure again with more details
                    with open(self.failed_batches_file, 'a') as f:
                        self.log_failed_batch(-1, batch, e)
                    
                    sleep(5)
                
                # update progress bar
                bar()

        if failed_retries > 0:
            print(f"✓ Retry complete with {failed_retries} failures remaining. Check {self.failed_batches_file}.")
        else:
            # Delete the failed batches file if all retries succeeded
            self.failed_batches_file.unlink(missing_ok=True)
            print("All failed batches successfully retried!")

        print("Failed batch retries complete.")


if __name__ == "__main__":
    import argparse
    from os import getenv
    from dotenv import load_dotenv

    load_dotenv()
    api_key_str = getenv("MISTRAL_API_KEY")
    
    if not api_key_str:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    api_key_str = api_key_str.strip()

    # Setup Mistral model and embeddings
    embeddings = MistralAIEmbeddings(api_key=api_key_str, model="mistral-embed")

    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory", "--input", "-i",
        type=str,
        default="../data/chunked_documents/",
        help="Directory containing chunked JSONL documents"
    )
    parser.add_argument(
        "--vectorstore_path", "--vectorstore",
        type=str,
        default="../chroma_db",
        help="Path to store the ChromaDB vector database"
    )
    parser.add_argument(
        "--collection_name", "--database_name",
        type=str,
        default="default_collection",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--logs_directory", "--log",
        type=str,
        default="../logs/",
        help="Directory to store logs like checkpoints and failed batches"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of documents to process in each batch"
    )
    parser.add_argument(
        "--delay_seconds",
        type=float,
        default=1,
        help="Seconds to wait between batches"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Enable full database rebuild, otherwise only process new/modified files"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry embedding for previously failed batches"
    )
    parser.add_argument(
        "--retry_only",
        action="store_true",
        help="Retry only the failed batches from the log"
    )

    args = parser.parse_args()

    # Setup and create ChromaDB
    creator = CreateChromaDB(
        embeddings=embeddings,
        collection_name=args.collection_name,
        chunked_docs_dir=Path(args.input_directory),
        chroma_db_dir=Path(args.vectorstore_path),
        checkpoint_file=Path(args.logs_directory)/"embedding_checkpoint.json",
        failed_batches_file=Path(args.logs_directory)/"failed_batches.txt"
    )

    # Use the new method with checkpointing, retry logic, and incremental updates
    # Set rebuild=False and resume=True (default) to only process new or modified documents
    # Set rebuild=True and resume=False to rebuild the entire database from scratch
    # If rebuild=False and resume=False, it will reprocess new/modified documents from the start
    # If rebuild=True and resume=True, it will resume full processing from the last checkpoint
    if args.retry_only:
        creator.retry_failed_batches(batch_size=args.batch_size, 
                                     delay_seconds=args.delay_seconds)
    else:

        creator.embed_and_store(batch_size=args.batch_size, 
                                delay_seconds=args.delay_seconds, 
                                resume=args.resume, 
                                rebuild=args.rebuild)
        
        # Retry failed batches if needed
        if args.retry:
            creator.retry_failed_batches()
    
    print("ChromaDB creation complete.")
