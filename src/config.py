from functools import lru_cache
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
# pyrefly: ignore [missing-import]
from pydantic import Field, model_validator

class Settings(BaseSettings):
    '''
    a class that is used to store configuration settings for the application
    
    '''

    #giving instruction to Pydantic on how it should behave when it tries to read the config/ load the variables
    model_config = SettingsConfigDict(env_file = (".env", "env"), env_prefix = "RAG_", extra = "ignore")
    
    data_dir: Path = Path("../data")
    storage_dir: Path = Path("../storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    #object Field is used to store data for a variable, automatically check for constraints like ge(greater than or equal)
    chunk_size: int = Field(default = 1000, ge = 100)
    chunk_overlap: int = Field(default = 150, ge = 0)
    top_k: int = Field(default = 5, ge = 1, le = 64)
    embedding_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"

    #---- llm provider configurations ----
    llm_provider: Literal["gemini", "vllm", "ollama"] = "ollama"  # Used for answering RAG
    judge_llm_provider: Literal["gemini", "vllm", "ollama", "ollama_judge"] = "ollama_judge"  # Used as LLM-as-a-judge
    llm_temperature: float = Field(default = 0.1, ge = 0.0, le = 0.2)

    #---- detailed llm configurations ----
    #Ollama
    ollama_model: str = "qwen2.5:3b"        # Model used for answering RAG
    ollama_judge_model: str = "phi3.5:3.8b" # Model used when judging (if judge_llm_provider='ollama_judge')

    #Gemini
    #running through cloud
    gemini_model: str = "gemini-2.5-flash"
    google_api_key: str| None = Field(default = None, validation_alias = "GOOGLE_API_KEY")

    #vLLM
    #running in my computer so dont need api key
    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "Qwen/Qwen2.5-3B-Instruct"

    #----retriever and study tools configurations-----
    #take 10 chunks to a batch and map first and then use this map to reduce
    summarize_batch_size: int = Field(default = 10, ge = 1)
    #take k chunks about a specific topic to generate summary
    summarize_retrieval_k: int = Field(default = 12, ge = 1, le = 128)
    #take k chunks about a specific topic to generate flashcards and quiz
    #Qwen 2.5 3B can reliably produce structured JSON with up to ~8 chunks; more causes empty or malformed output
    generation_retrieval_k: int = Field(default = 8, ge = 1, le = 128)
    #default quiz number generated
    quiz_default_count: int = Field(default = 8, ge = 1, le = 50)
    #default flashcards number generated
    flashcards_default_count: int = Field(default = 15, ge = 1, le = 100)

    api_url: str = "http://localhost:8000"
    
    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        #logic when chunking
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
            
        #logic when gemini is chosen
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when llm_provider='gemini'.")
            
        return self

#LRU cache for storing the settings
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()