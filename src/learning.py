
import logging
import time

# pyrefly: ignore [missing-import]
from pydantic import ValidationError


from rag import format_citations
from schemas import Summary
from llm import invoke_llm
from config import settings
import json

logger = logging.getLogger(__name__)

MAX_LLM_RETRIES = 3
from rag import fetch_all_chunks
from rag import retrieve
from rag import render_prompt
from schemas import QuizItem
from schemas import QuizSet
from schemas import FlashcardSet
from schemas import Flashcard



#summary templates  
SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"   
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"         
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"   

#quiz and flashcards templates
QUIZ_TEMPLATE = "quiz.jinja2"                       
FLASHCARDS_TEMPLATE = "flashcards.jinja2"           



def _resolve_target(document, query, filters, k, retrieval_k):
    '''
    this function acts like a data dispatcher
    scope: corpus, document, query, filter
    target: None, filename, query, specific pages
    Args: document, query, filters, k, retrieval_k
    Return: chunks, scope, target
    '''
    #using dict(filters or {}) meaning: if user doesnt choose any filter -> filters = None -> if None["filename"] -> error
    # so dict(filters or {}) meaning we create a copy of original input, if input is None -> return empty dict
    effective_filters = dict(filters or {})
    #assign document to effective_filters if it exists
    if document:
        effective_filters["filename"] = document
    
    if query:
        chunks = retrieve(query, top_k = k or retrieval_k, filters = effective_filters)
        return chunks, "query", query

    # Cap chunks to avoid overwhelming the LLM with too-long prompts
    max_chunks = k or retrieval_k

    if effective_filters:
        chunks = fetch_all_chunks(filters = effective_filters)[:max_chunks]
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target
    
    chunks = fetch_all_chunks()[:max_chunks]
    return chunks, "corpus", None

#------ Summary ------
'''
for summary, system supports 2 mode: 
    1. if chunks are not too large, system reads all and summarizes using prompt: summary_single.jinja2 
    2. if chunks are too large, system Map-Reduce: summaries each chunk then merge them
    Note: it still generates citations for each chunk

'''
def _parse_json(text):
    '''
    parse the raw text from llm into JSON object for the application to use
    '''
    cleaned = text.strip()
    # Guard against empty LLM response
    if not cleaned:
        raise json.JSONDecodeError("LLM returned empty response", "", 0)
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        # Remove opening fence (e.g. ```json or ```)
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    # Also handle triple single-quotes
    if cleaned.startswith("'''"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("'''"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    obj = json.loads(cleaned)

    if not isinstance(obj, (dict, list)):
        raise RuntimeError("Expected JSON object or array.")
    return obj


def _invoke_llm_json(prompt, retries=MAX_LLM_RETRIES):
    '''
    Invoke the LLM and parse the JSON response with automatic retry logic.
    Small local models (e.g., Qwen 2.5 3B) sometimes return empty or malformed
    responses, especially with long prompts. This wrapper retries up to N times
    before raising an error.
    '''
    last_error = None
    for attempt in range(1, retries + 1):
        raw = invoke_llm(prompt)
        try:
            return _parse_json(raw)
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            logger.warning(
                "LLM returned invalid JSON (attempt %d/%d): %s | Raw (first 200 chars): %r",
                attempt, retries, e, raw[:200] if raw else "<empty>"
            )
            if attempt < retries:
                time.sleep(1)  # brief pause before retry
    raise RuntimeError(
        f"LLM failed to return valid JSON after {retries} attempts. "
        f"Last error: {last_error}"
    )

def _validate_summary_payload(payload: dict):
    """
    This function validate and extract summary data from LLM JSON payload
    1. check if payload is dict
    2. extract and check summary text
    3. extract and format keypoints
    """
    #1. check if payload is dict
    if not isinstance(payload, dict):
        raise ValueError("Payload from LLM is not a valid Dictionary (JSON object).")
    
    #2. extract and check summary text
    summary_text = payload.get("summary")
    if not summary_text or not isinstance(summary_text, str):
        raise ValueError("Payload bị thiếu trường 'summary' hoặc không phải là chuỗi văn bản.")
        
    # 3. extract and format keypoints
    key_points = payload.get("key_points", []) 
    if not isinstance(key_points, list):
        #if llm returns non-list data , convert it into list
        key_points = [] 
    else:
        #convert every element into string
        key_points = [str(kp).strip() for kp in key_points if str(kp).strip()]
        
    # return (summary_text, key_points)
    return summary_text.strip(), key_points

def summarize(document = None, query = None, filters = None, k = None):
    '''
    This function is used to generate a summary of the given chunks
    Payload means a container to carry JSON data that AI returns according to instruction
    Workflow: retrieve chunk with filters -> feed chunk into prompt as context -> requires LLM to return structured answer 
    -> validate and citations
    Args: 
        document: the document to summarize
        query: the query to summarize
        filters: the filters to apply to the query
        k: the number of chunks to summarize
    
    Returns: 
        Summary object predefined in schemas.py
    '''
    chunks, scope, target = _resolve_target(document, query, filters, k, settings.summarize_retrieval_k)

    # Guard: if no chunks found, return a helpful message instead of crashing
    if not chunks:
        return Summary(
            summary="No content found for the selected document(s). Please ensure the document has been uploaded and indexed.",
            scope=scope, target=target, key_points=[], citations=[], chunks=[]
        )

    if len(chunks) <= settings.summarize_batch_size:
        #handle the case when chunks is small, just summarize it once
        prompt = render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks = chunks)
        payload = _invoke_llm_json(prompt)
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            chunk_batch = chunks[start: start + settings.summarize_batch_size]
            prompt = render_prompt(SUMMARY_MAP_TEMPLATE, chunks = chunk_batch)
            payload = _invoke_llm_json(prompt)
            summary_text, key_points = _validate_summary_payload(payload)
            partials.append({"summary":summary_text, "key_points":key_points})

            #merge partial summary together
        payload = _invoke_llm_json(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials = partials))
        summary_text, key_points = _validate_summary_payload(payload)
    
    return Summary(
        summary = summary_text,
        scope = scope,
        target = target,
        key_points = key_points,
        citations = format_citations(chunks),
        chunks = chunks
    )


def _validate_items(payload, key, model_class, dedup_field, label, valid_markers):
    '''
    Args:
        payload: raw payload from LLM 
        key: key to extract items from payload
        model_class: Pydantic model class for validation
        dedup_field: field used to deduplicate items
        label: human-friendly label for logging
        valid_markers: set of valid source markers
    
    Return:
        dict_object: dictionary containing valid items, source markers, and citations
        
    '''
    #take items from payload
    raw_items = payload.get(key)
    #using set to check duplicate in O(1)
    items, seen = [], set()

    #parse and remove trash (Pydantic validation)
    #try to parse raw into Pydantic mould (example: check if quiz has 4 options)
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError as e:
            continue
    
        #deduplication
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        #if norm already seen -> skip
        if not norm or norm in seen:
            continue
        seen.add(norm)
        #add marker validation
        markers = [m for m in item.source_markers if m in valid_markers]
        #using model_copy() which allows duplicating model instance to update fields without running validation again
        items.append(item.model_copy(update = {"source_markers": markers}))

    if not items:
        raise RuntimeError(f"No valid {label} found in LLM response.")
    return items



def generate_quiz(document = None, query = None, filters = None, count = None, k = None):
    '''
    Args:
        document: the document to generate quiz from
        query: the query to generate quiz from
        filters: the filters to apply to the query
        count: the number of quiz to generate
        k: the number of chunks to generate quiz from
    
    Returns:
        QuizSet object predefined in schemas.py
    '''

    chunks, scope, target = _resolve_target(document, query, filters, k, settings.generation_retrieval_k)

    # Guard: if no chunks found, return empty quiz instead of crashing
    if not chunks:
        return QuizSet(scope=scope, target=target, items=[], chunks=[], citations=[])

    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(QUIZ_TEMPLATE, chunks = chunks, count = n)
    payload = _invoke_llm_json(prompt)

    quiz_items = _validate_items(payload, "quiz", QuizItem, "question", "quiz items", valid_markers)

    return QuizSet(
        scope = scope, 
        target = target,
        chunks = chunks,
        items = quiz_items,
        citations = format_citations(chunks)
    )


#------- Flashcards -------
def generate_flashcards(document=None, query=None, filters=None, count=None, k=None):
    chunks, scope, target = _resolve_target(document, query, filters, k, settings.generation_retrieval_k)

    # Guard: if no chunks found, return empty flashcard set instead of crashing
    if not chunks:
        return FlashcardSet(scope=scope, target=target, cards=[], chunks=[], citations=[])

    n = count or settings.flashcards_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload = _invoke_llm_json(prompt)
    
    cards = _validate_items(payload,"cards", Flashcard,"front","flashcards",valid_markers)
    
    return FlashcardSet(scope=scope, target=target, cards=cards, chunks=chunks, citations=format_citations(chunks))

    
    
    

        
    


    

    
        





    
