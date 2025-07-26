import os
import json
import logging
import argparse
from rag_pipeline import load_mimic_notes, build_or_load_faiss_index, rag_health_recommend

def setup_logging(output_dir: str):
    """Set up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "orchestrator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Personalized RAG Health Recommender Orchestrator")
    parser.add_argument("--mimic_path", type=str, default="data/mimic-iii-demo/NOTEEVENTS.csv", help="Path to MIMIC-III notes CSV")
    parser.add_argument("--max_notes", type=int, default=10000, help="Maximum number of notes to process for the index")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and index")
    parser.add_argument("--eval_dataset", type=str, default="evaluation_dataset.json", help="Path to the personalized query dataset (JSON)")
    return parser.parse_args()

def main():
    """Main function to orchestrate the personalized RAG pipeline."""
    args = parse_args()
    setup_logging(args.output_dir)
    
    logging.info("🚀 Starting Personalized RAG Health Recommender...")
    
    # 1. Load data and build/load index
    docs, titles = load_mimic_notes(args.mimic_path, args.max_notes)
    index_path = os.path.join(args.output_dir, "faiss_index_biobert.bin")
    faiss_index = build_or_load_faiss_index(docs, index_path)
    
    # 2. Load personalized queries
    try:
        with open(args.eval_dataset, "r") as f:
            queries_data = json.load(f)
        logging.info(f"Loaded {len(queries_data)} queries from {args.eval_dataset}")
    except Exception as e:
        logging.error(f"Failed to load queries from {args.eval_dataset}: {e}")
        return

    # 3. Generate answers for each query
    rag_outputs = []
    for item in queries_data:
        query = item["query"]
        context = item["patient_context"]
        try:
            result = rag_health_recommend(query, context, args.top_k, faiss_index, docs, titles)
            rag_outputs.append(result)
            logging.info(f"Generated answer for query: '{query}'")
        except Exception as e:
            logging.error(f"Failed to generate answer for query '{query}': {e}")
    
    # 4. Save RAG outputs
    output_path = os.path.join(args.output_dir, "rag_personalized_output.json")
    with open(output_path, "w") as f:
        json.dump(rag_outputs, f, indent=2)
    logging.info(f"✅ Saved all personalized RAG outputs to {output_path}")

if __name__ == "__main__":
    main()
