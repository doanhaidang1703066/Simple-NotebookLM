import collections
from functools import lru_cache
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_qdrant import QdrantVectorStore
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http import models as qmodels
from config import settings


@lru_cache(maxsize = 1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = settings.embedding_model,
        encode_kwargs = {'normalize_embeddings': True}
    )

@lru_cache(maxsize = 1)
def get_client():
    '''
    this function creates a connection to the local Qdrant vector database
    it checks if the storage_dir exists, if not, it creates it, using exist_ok = True prevents the program from crashing
    lru_cache ensures that the client is created only once and reused throughout the application to avoid unnecessary overhead.
    '''
    settings.storage_dir.mkdir(parents = True, exist_ok = True)
    return QdrantClient(path = str(settings.storage_dir))

@lru_cache(maxsize = 1)
def get_vector_store(collection_name = None):
    return QdrantVectorStore(
        collection_name = collection_name or settings.qdrant_collection,
        embedding = get_embeddings(),
        client = get_client()
    )

#----- code for setting up collections and indices --------
'''
create a payload schema for fields that users often search for to make it easier for the index:
including document_id, filename, page

index is a data structure used to speed up the retrieval process
in this file we have to create a new collection or use the already-existed collection
a collection is created with collection_name and vectors_config
vectors_config includes: vector_size and distance metric

qmodels.PayloadSchemaType.KEYWORD is used to allow for filtering using the exact keyword
qmodels.PayLoadSchemaType.INTEGER is used to allow for filtering using the exact integer
then for each field we check if a payload schema is applied to it
'''

INDEXES_PAYLOAD_FIELDS = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER
}

def ensure_collection(recreate = False, collection_name = None):
    client = get_client()
    name = collection_name or settings.qdrant_collection
    
    #check if exists
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    
    if not exists: #the database requires you to specify the dim vector 
        dim = len(get_embeddings().embed_query("dimension probe"))
        #creating a new collection require vectors_config: distance type and vector dim
        client.create_collection(
            collection_name = name,
            vectors_config = qmodels.VectorParams(
                size = dim,
                distance = qmodels.Distance.COSINE
            ))
            
            

    #check if payload schema is applied to every field
    payload_schema = client.get_collection(name).payload_schema or {}
    for field, schema in INDEXES_PAYLOAD_FIELDS.items():
        if payload_schema.get(field) is None:
            client.create_payload_index(name, field_name = field, field_schema = schema)

        
    
