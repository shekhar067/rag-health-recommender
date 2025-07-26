
import os
import logging
import argparse
from typing import Dict, List, Tuple
from rag_pipeline import build_rag_pipeline, generate_answer
from rag_evaluate import evaluate_pipeline

def setup_logging(output_dir: str):
    """Set up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "orchestrator.log")),
            logging.StreamHandler()
        ]
    )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAG Health Recommender Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo/NOTEEVENTS.csv",
                        help="Path to MIMIC-III notes CSV")
    parser.add_argument("--diagnoses_path", type=str, default="data/mimic-iii-demo/DIAGNOSES_ICD.csv",
                        help="Path to MIMIC-III diagnoses CSV")
    parser.add_argument("--max_notes", type=int, default=10000,
                        help="Maximum number of notes to process")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of top documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--query_file", type=str, default="queries.txt",
                        help="Path to query file")
    return parser.parse_args()

def main():
    """Main function to orchestrate the RAG pipeline and evaluation."""
    args = parse_args()
    setup_logging(args.output_dir)
    
    logging.info("Starting RAG Health Recommender...")
    
    # Build RAG pipeline
    logging.info("Building RAG pipeline...")
    rag_pipeline, faiss_index = build_rag_pipeline(
        args.mimic_path,
        max_notes=args.max_notes,
        index_path=os.path.join(args.output_dir, "faiss_index.bin")
    )
    
    # Load queries
    try:
        with open(args.query_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(queries)} queries from {args.query_file}")
    except Exception as e:
        logging.error(f"Failed to load queries: {e}")
        raise
    
    # Generate answers
    rag_outputs = []
    for query in queries:
        try:
            answer, retrieved_docs = generate_answer(query, rag_pipeline, faiss_index, args.top_k)
            rag_outputs.append({
                "query": query,
                "answer": answer,
                "retrieved_docs": retrieved_docs
            })
            logging.info(f"Generated answer for query: {query}")
        except Exception as e:
            logging.error(f"Failed to generate answer for query '{query}': {e}")
    
    # Save RAG outputs
    output_path = os.path.join(args.output_dir, "rag_output.json")
    try:
        with open(output_path, "w") as f:
            json.dump(rag_outputs, f, indent=2)
        logging.info(f"Saved RAG outputs to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save RAG outputs: {e}")
    
    # Evaluate pipeline
    try:
        results = evaluate_pipeline(
            args.mimic_path,
            args.diagnoses_path,
            num_samples=50,
            output_dir=args.output_dir
        )
        logging.info("Evaluation completed")
        logging.info(f"Results: {results}")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
