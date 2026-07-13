
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator
#Basemodel is used as a base class for data parsing, validation and configuration
from typing import Literal 


from filters import MetadataFilter


#-------- Retrieval Data --------
class ChunkMetadata(BaseModel):
    '''
    this class includes source information of each chunk including:
    - document_id
    - filename
    - source (directory)
    - page
    - chunk_id
    - section
    '''

    document_id: str
    filename: str
    source: str
    page: int 
    chunk_id: str
    section: str|None


class RetrievedChunk(BaseModel):
    '''
    this class is used to store the content and the source of retrieved chunk
    '''
    metadata: ChunkMetadata
    text: str
    score: float

class Citation(BaseModel):
    '''
    this class is used to store source_marker like [S1], [S2] and source_index
    this class is used to normalize the citation of each retrieved chunk to be consistent like [S1], [S2]
    '''
    source_marker: str
    source_index: int

    filename: str
    page: int
    section: str|None
    chunk_id: str|None

# ------ Output Data --------
class RagAnswer(BaseModel):
    '''
    this class is used to store the information of the final Answer including:
    citations
    source chunks
    question
    answer
    '''
    question:str
    answer: str
    citations: list[Citation] = Field(default_factory = list)
    chunks: list[RetrievedChunk] = Field(default_factory = list)
    

#------- Tools for Learning Data ------
class Summary(BaseModel):
    '''
    this class is responsible for containerizing the answer after the LLM summarizes the work
    scope:
    +, scope of corpus would be all documents
    +, scope of document would be all documents
    +, scope of query would be according to the query
    +, scope of filter would be the specific section like from page 5 to page 10
    '''
    summary: str 
    scope: Literal["corpus", "document", "query", "filter"]
    #target could be the filename if scope is document or a query string if scope is query or the specific section if scope is filter
    target: str|None 
    #key_points are parsed as the list of strings in order to help the UI showing key points of summary instead of long string summary
    key_points: list[str] = Field(default_factory = list)
    citations: list[Citation] = Field(default_factory = list)
    chunks: list[RetrievedChunk] = Field(default_factory = list)

class QuizItem(BaseModel):
    question: str
    options: list[str] = Field(min_length = 4, max_length = 4)
    correct_index: int #from 0 to 3
    explanation: str
    #source_markers is the list of citation marker like [S1], [S2] that are relevant to this question
    source_markers: list[str] = Field(default_factory = list)
    difficulty: str|None = None
    topic: str|None = None 

    @model_validator(mode = "after")
    def _validate_correct_index(self) -> "QuizItem":
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError("correct_index out of range")
        return self

class QuizSet(BaseModel):
    scope: Literal["query", "filter", "document", "corpus"]
    target: str|None = None
    items: list[QuizItem] = Field(default_factory = list)
    citations: list[Citation] = Field(default_factory = list)
    chunks: list[RetrievedChunk] = Field(default_factory = list)

class Flashcard(BaseModel):
    front: str
    back: str
    hint: str|None = None
    topic: str|None = None
    source_markers: list[str] = Field(default_factory = list)

class FlashcardSet(BaseModel):
    scope: Literal["query", "filter", "document", "corpus"]
    target: str|None = None
    cards: list[Flashcard] = Field(default_factory = list)
    citations: list[Citation] = Field(default_factory = list)
    chunks: list[RetrievedChunk] = Field(default_factory = list)


#------ API data --------
class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)

class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    pass
    
class DocumentInfo(BaseModel):
    filename: str


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int

