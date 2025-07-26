import subprocess
import argparse
import logging
import os
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outputs/orchestrator.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator for RAG Health Recommender")
    parser.add_argument("--mimic_path", default="data/mimic-iii/NOTEEVENTS.csv", help="Path to MIMIC-III NOTEEVENTS.csv")
    parser.add_argument("--diagnoses_path", default="data/mimic-iii/DIAGNOSES_ICD.csv", help="Path to MIMIC-III DIAGNOSES_ICD.csv")
    parser.add_argument("--max_notes", type=int, default=10000, help="Max notes to process")
    parser.add_argument("--top_k", type=int, default=2, help="Number of documents to retrieve")
    parser.add_argument("--index_path", default="outputs/faiss_index.bin", help="Path to save/load FAISS index")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save results")
    parser.add_argument("--query_file", default="queries.txt", help="Path to file with queries (one per line)")
    return parser.parse_args()

def run_script(script_name: str, args: List[str]) -> bool:
    """Run a Python script with given arguments and return success status."""
    try:
        logging.info(f"Running {script_name} with args: {args}")
        result = subprocess.run(
            ["python", script_name] + args,
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"{script_name} completed successfully. Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e.stderr}")
        return False

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "rag_output.json")
    eval_output = os.path.join(args.output_dir, "eval_results.json")
    
    # Step 1: Run rag_pipeline.py
    pipeline_args = [
        "--mimic_path", args.mimic_path,
        "--max_notes", str(args.max_notes),
        "--top_k", str(args.top_k),
        "--index_path", args.index_path,
        "--output", output_json,
        "--query_file", args.query_file
    ]
    
    logging.info("Starting RAG pipeline...")
    if not run_script("rag_pipeline.py", pipeline_args):
        logging.error("RAG pipeline failed. Aborting.")
        return
    
    # Step 2: Run rag_evaluate.py
    eval_args = [
        "--mimic_path", args.mimic_path,
        "--diagnoses_path", args.diagnoses_path,
        "--output", eval_output
    ]
    logging.info("Starting evaluation...")
    if not run_script("rag_evaluate.py", eval_args):
        logging.error("Evaluation failed.")
        return
    
    logging.info("Orchestration complete. Results saved in %s", args.output_dir)

if __name__ == "__main__":
    main()