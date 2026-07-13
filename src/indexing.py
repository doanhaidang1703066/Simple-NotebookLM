'''
This file is used to prepare for the loading, chunking, and saving into vector database pipeline
First we use the PyPDFLoader --> Recursive splitter --> Embedding --> Saving/ Qdrant - Vector Store
'''
import uuid
from store import get_vector_store, ensure_collection
import hashlib
from pathlib import Path
from collections import defaultdict

#---- import libraries for extract and transform pipeline of ETL from Langchain -----
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

#--- import modules from our projects -----
from config import settings
from schemas import ChunkMetadata, DocumentInfo


def discover_pdfs():
    '''
    search through data folder, return lists of paths of pdfs that will be used to build the pipeline.
    '''
    directory_path = settings.data_dir
    pdf_paths = list(directory_path.glob("*.pdf"))
    
    return pdf_paths




#---- build functions to handle loading pdf files and assign metadata with document_id, filename, source, page and section ----
def _document_id(path) -> str:
    """Generate a document ID from a file path and its size in byte"""
    raw = f"{path.name}{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def _chunk_id(doc_id, page, index) -> str:
    return f"{doc_id}:{page}:{index}"

def _load_pdf(path):

    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)

    for doc in pages:
        page_number = doc.metadata.get("page", 0) + 1 # +1 to make page number human-readable (1-indexed)
        doc.metadata = {
            "document_id": doc_id, 
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section")
        }

    return pages

def _splitter(chunk_size = None, chunk_overlap = None):
    """
    build the recursive splitter function
    """
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size or settings.chunk_size,
        chunk_overlap = chunk_overlap or settings.chunk_overlap,
        separators = ["\n\n", "\n", ".", " ", ""],
        keep_separator = False
    )


def build_chunks(pdf_paths, chunk_size = None, chunk_overlap = None, chunker = None):

    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(path))
    
    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)

    #initialize a defaultdict to keep track of the number of chunks for each document.
    #when it sees a new document_id, it automatically initializes the count to 0
    #if page 1 has 3 chunks, when adding new chunks on different pages, it should start counting from the next number after the last count on that specific document.
    per_doc_counter = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1

        meta = ChunkMetadata(
            document_id = doc_id,
            filename = chunk.metadata["filename"],
            source = chunk.metadata["source"],
            page = chunk.metadata["page"],
            chunk_id = _chunk_id(doc_id, chunk.metadata["page"], idx),
            section = chunk.metadata.get("section")
        )
        
        chunk.metadata = meta.model_dump()

    return chunks

def index_chunks(chunks, collection_name = None):
    '''
    this function add chunks into database after checking if there is any chunk in the database
    Args:
        chunks: 
        collection_name: 
    
    Returns:
        len(chunks)
    '''

    if not chunks:
        return 0
    #create chunk id to uniquely identify each chunk in the database
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name = collection_name).add_documents(documents = chunks, ids = ids)
    return len(chunks)
    
def ingest(recreate = False, collection_name = None, chunk_size = None, chunker = None, chunk_overlap = None):
    #ensure the collection and index exists
    pdf_paths = discover_pdfs()
    ensure_collection(
        recreate = recreate,
        collection_name = collection_name
    )
    
    chunks = build_chunks(
        pdf_paths = pdf_paths,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        chunker = chunker
    )

    return index_chunks(chunks, collection_name)

def save_and_ingest(file_bytes, filename):
    safe_name = Path(filename).name
    dest = settings.data_dir / safe_name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    return {"filename": safe_name, "chunks_indexed": index_chunks(chunks)}
    

from schemas import DocumentInfo

def list_documents():
    """
    Scans the data directory and returns a list of DocumentInfo objects
    representing all available PDF files.
    """
    # Uses the discover_pdfs() function you already wrote!
    pdf_paths = discover_pdfs() 
    
    document_list = []
    for path in pdf_paths:
        # Extract just the filename (e.g., "Logistics.pdf")
        doc_info = DocumentInfo(filename=path.name)
        document_list.append(doc_info)
        
    return document_list

    


    

