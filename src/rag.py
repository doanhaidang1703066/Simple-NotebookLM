import collections
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from schemas import ChunkMetadata, RagAnswer, Citation, RetrievedChunk
from config import settings
from store import get_vector_store, get_client
from filters import filters_to_qdrant
from llm import invoke_llm
'''
RAG Pipeline:
define function retrieve , fetch_all_chunks, jinjia, render_prompt, format_citations, and finally answer
'''
# directory path to store all prompt files
PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"
def retrieve(query, top_k = None, filters = None, collection_name = None):
    '''
    Args: 
    - query: the query text
    - top_k: number of chunks to retrieve
    - filters: filters to apply to the retrieval
    - collection_name: name of the collection to retrieve from
    Return: retrieved_chunks in format List[RetrievedChunk]
    '''
    hits = get_vector_store(collection_name).similarity_search_with_score(
        query = query,
        k = top_k or settings.top_k,
        filter = filters_to_qdrant(filters)
    )
    
    # **doc.metadata meaning we unzip the metadata dictionary
    return [
        RetrievedChunk(
            metadata = ChunkMetadata(**doc.metadata),
            text = doc.page_content,
            score = float(score) 
        )
        for doc, score in hits
    ]

def scroll_all(collection_name, filter=None, batch_size=100):
    """
    Get all chunks from Qdrant by batch size to avoid overload RAM
    using yield so that it generates instead of return, only return when iterating because it would cause ram overload when trying to return
    all at once so generate in batch
    """
    client = get_client()
    offset = None
    while True:
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter,
            limit=batch_size,
            offset=offset,
            with_payload=True
        )
        if not records:
            break
        yield records  # return each page to fetch_all_chunks
        if next_page_offset is None:
            break
        offset = next_page_offset


def fetch_all_chunks(collection_name = None, filters = None):
    '''
    Fetch all chunks from the collection using filters not using semantic search
    This function used the scroll_all to fetch all chunkss

    scroll_all(name, filter = filters_to_qdrant(filters)) uses to get all points in a page-by-page manner
    '''
    name = collection_name or settings.qdrant_collection
    retrieved_chunks = []

    for page in scroll_all(name, filter = filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta, text = payload.get("metadata") or {}, payload.get("page_content") or ""

            if meta and text:
                retrieved_chunks.append(RetrievedChunk(metadata = ChunkMetadata(**meta), text = text, score = 0.0))

    #return sorted retrieved_chunks according to filename, page, chunk_id
    return sorted(
        retrieved_chunks, key = lambda r: (
            r.metadata.filename,
            r.metadata.page,
            int(r.metadata.chunk_id.rsplit(":", 1)[-1])
        )
    )




@lru_cache(maxsize = 1)
def _jinja_env():
    return Environment(
        loader = FileSystemLoader(str(PROMPTS_DIR)),
        autoescape = False,
        undefined = StrictUndefined,
        trim_blocks = True,
        lstrip_blocks = True,
    )

def render_prompt(template_name, **context):
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks):
    return [
        Citation(
            source_index = i,
            source_marker = f"S{i}",
            filename = c.metadata.filename,
            page = c.metadata.page,
            section = c.metadata.section,
            chunk_id = c.metadata.chunk_id
        )
        for i, c in enumerate(chunks, start = 1) #1-based indexing
    ]

def answer(question, k = None, filters = None, collection_name = None):
    chunks = retrieve(question, top_k = k, filters = filters, collection_name = collection_name)
    
    
    if not chunks:
        return RagAnswer(
            question = question,
            answer = "Not enough information in context to answer this question"
        )
    prompt = render_prompt(ANSWER_TEMPLATE, question = question, chunks = chunks)
    text = invoke_llm(prompt)

    return RagAnswer(
        question = question,
        answer = text.strip(),
        chunks = chunks,
        citations = format_citations(chunks)
    )

    


        

    








