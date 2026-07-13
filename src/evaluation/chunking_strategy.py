from dataclasses import dataclass, field
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag import answer

#------RECURSIVE CHARACTER TEXT SPLITTERS --------
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200)
]


#dataclass automatically adds __init__ and __repr__ to the class
@dataclass(frozen = True) #this allows the class to be immutable
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict[str, object]
@dataclass(frozen = True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)

    def  _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators = self.separators or DEFAULT_SEPARATORS,
            is_separator_regex = False
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)


#----- SEMANTIC TEXT SPLITTERS -----
# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
# pyrefly: ignore [missing-import]
from langchain_core.embeddings import Embeddings

#split the text based on percentile, std, interquartile
_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile")]

@dataclass(frozen = True)
class SemanticChunker:
    """ Langchain Wrapper for SemanticChunker"""
    embeddings: Embeddings
    breakpoint_type: str = "percentile"

    def _splitter(self) -> SemanticChunker:
        return SemanticChunker(breakpoint_type = self.breakpoint_type, embeddings = self.embeddings)
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)

