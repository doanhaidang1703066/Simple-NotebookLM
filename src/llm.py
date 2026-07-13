from functools import lru_cache

# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI

from config import settings

def _build_ollama(model_name: str | None = None):
    return ChatOllama(
        model = model_name or settings.ollama_model,
        temperature = settings.llm_temperature
    )

def _build_gemini():
    return ChatGoogleGenerativeAI(
        model = settings.gemini_model,
        temperature = settings.llm_temperature,
        api_key = settings.google_api_key
    )

def _build_vllm():
    return ChatOpenAI(
        model = settings.vllm_model,
        openai_api_key = settings.vllm_api_key,
        openai_api_base = settings.vllm_api_base,
        temperature = settings.llm_temperature
    )

@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    """
    this function returns a ChatOpenAI instance that is configured to use the appropriate model and API key
    the lru_cache decorator ensures that the instance is created only once and reused throughout the application.
    """
    provider = provider or settings.llm_provider

    if provider == "ollama":
        return _build_ollama(settings.ollama_model)
    elif provider == "ollama_judge":
        return _build_ollama(settings.ollama_judge_model)
    elif provider == "gemini":
        return _build_gemini()
    elif provider == "vllm":
        return _build_vllm()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def invoke_llm(prompt):
    response = get_llm().invoke([HumanMessage(content = prompt)])
    return response.content if isinstance(response, str) else str(response.content)
    
