# --- Compatibility shim for ragas 0.4.3 + langchain-community 0.4.x ---
# ragas hardcodes `from langchain_community.chat_models.vertexai import ChatVertexAI`
# but langchain-community 0.4.x removed that module. We create a fake
# module at the old path, backed by langchain-google-vertexai.
import sys, types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI  # type: ignore
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = ChatVertexAI  # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = _shim
# --- End shim ---

# pyrefly: ignore [missing-import]
from ragas import evaluate
# pyrefly: ignore [missing-import]
from ragas.embeddings import LangchainEmbeddingsWrapper
# pyrefly: ignore [missing-import]
from ragas.llms import LangchainLLMWrapper
# pyrefly: ignore [missing-import]
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# pyrefly: ignore [missing-import]
from ragas.run_config import RunConfig 
# pyrefly: ignore [missing-import]
from datasets import Dataset
from typing import Callable

from llm import get_llm
from schemas import RagAnswer
from store import get_embeddings

#wrap the llm + embeddings --> get_ragas_metrics (metrics configuration) --> preparing the dataset -> evaluate

def get_ragas_metrics(llm, embeddings):
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_recall.llm = llm
    

    context_precision.embeddings = embeddings
    context_recall.embeddings = embeddings
    
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return metrics

def run_evaluation(
    test_cases: list[dict[str, str]], *,
    answer_fn: Callable[[str], RagAnswer],
    llm_provider: str|None = None,
    timeout_s: int = 180, 
    max_retries: int = 3, 
    max_workers: int = 4
    ):

    '''
    Args: 
    - Test cases: dict[str, list] = {"user_input", "response", "retrieved_context", reference"}
    - answer_fn: function to generate answer
    - llm_provider: provider of the LLM
    - timeout_s: timeout in seconds
    - max_retries: max retries
    - max_workers: max workers

    Return: 
        dict[str, Any]: Evaluation results with metrics.
    '''

    # 1) Generate Responses: iterate through all test cases and call answer_fn on the user input
    
    data: dict[str, list] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }

    for case in test_cases:
        rag_response = answer_fn(case["question"])
        data["user_input"].append(case["question"])
        data["response"].append(rag_response.answer)
        data["reference"].append(case["ground truth"])
        
        #retrieved context: we need to map it back from the chunks
        data["retrieved_contexts"].append([chunk.text for chunk in rag_response.chunks])

    # 2) Build Dataset: create a Dataset object for Ragas to use
    eval_dataset = Dataset.from_dict(data)

    llm = LangchainLLMWrapper(get_llm(llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)

    config = RunConfig(
        timeout = timeout_s,
        max_retries = max_retries,
        max_workers = max_workers
    )

    return evaluate(
        dataset = eval_dataset,
        metrics = metrics,
        llm = llm,
        embeddings = embeddings,
        run_config = config
    )


    






