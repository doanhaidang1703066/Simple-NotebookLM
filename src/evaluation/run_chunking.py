from pathlib import Path
from config import settings
from schemas import RagAnswer
from indexing import ingest
from rag import answer
from store import get_embeddings
from evaluation.ragas_evaluator import run_evaluation
from evaluation.chunking_strategy import (
    ChunkingStrategy, RecursiveChunker, SemanticChunker,
    _RECURSIVE_CONFIGS, _SEMANTIC_CONFIGS
)
from utils import write_json, summary_metrics 

def _evaluate_strategy(
    strategy: ChunkingStrategy, 
    output_dir: Path, 
    test_cases: list[dict]
) -> dict[str, object]:
    """
    Evaluates a specific chunking strategy by ingesting documents into a unique 
    Qdrant collection and running Ragas metrics against it.
    """
    # Create a unique collection name for this specific strategy
    collection_name = f"{settings.qdrant_collection}_{strategy.strategy_id}"
    
    # Ingest the documents using this specific chunker
    chunk_count = ingest(
        recreate=True, 
        collection_name=collection_name, 
        chunker=strategy.chunker
    )
    
    result_out: dict[str, object] = {
        "strategy_id": strategy.strategy_id,
        "chunk_count": chunk_count,
        "summary_metrics": {},
    }

    try:
        # Define the specific answer function that targets this strategy's collection
        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        # Run the Ragas evaluation using the configured judge LLM provider
        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider=settings.judge_llm_provider)
        
        # Convert results to pandas dataframe to calculate summary metrics
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)
        
    except Exception as exc:
        result_out["error"] = str(exc)

    # Save the results to a JSON file
    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    
    return result_out


# --- Execution Block ---
if __name__ == "__main__":
    import pandas as pd

    _SCRIPT_DIR = Path(__file__).resolve().parent
    df_tests = pd.read_csv(_SCRIPT_DIR / "Data-Benchmark-Rag.csv")
    test_cases = df_tests.to_dict('records')
    
    
    output_dir = Path("evaluation_results/chunking")
    output_dir.mkdir(parents=True, exist_ok=True)

    ALL_STRATEGIES = []
    
    # 3a. Add all Recursive Chunking strategies-Building-Simple-NotebookLM-2.pdf]
    for strategy_id, chunk_size, chunk_overlap in _RECURSIVE_CONFIGS:
        strategy = ChunkingStrategy(
            strategy_id=strategy_id,
            chunker=RecursiveChunker(chunk_size = chunk_size, chunk_overlap = chunk_overlap),
            params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )
        ALL_STRATEGIES.append(strategy)
        
    embeddings = get_embeddings() # Ensure this function is available 
    for strategy_id, breakpoint_type in _SEMANTIC_CONFIGS:
         strategy = ChunkingStrategy(
             strategy_id=strategy_id,
             chunker=SemanticChunker(embeddings=embeddings, breakpoint_type=breakpoint_type),
             params={"breakpoint_type": breakpoint_type}
         )
         ALL_STRATEGIES.append(strategy)

    # 4. The Main Loop: Execute evaluation for each strategy
    for strategy in ALL_STRATEGIES:
        print(f"Starting evaluation for strategy: {strategy.strategy_id}...")
        
        # Call the evaluation function we built earlier
        result = _evaluate_strategy(
            strategy=strategy, 
            output_dir=output_dir, 
            test_cases=test_cases
        )
        
        # Optional: Print a brief summary of the results
        if "error" in result:
             print(f"Error during {strategy.strategy_id}: {result['error']}")
        else:
             print(f"Completed {strategy.strategy_id}.")
             print(f"Chunks Indexed: {result.get('chunk_count', 0)}")
             print(f"Scores: {result.get('summary_metrics', {})}\n")
             
    print("All chunking evaluations complete! Check the output directory for JSON reports.")
