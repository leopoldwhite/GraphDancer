import argparse
import glob
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

# Ensure we can import from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.graph_constants import GRAPH_DEFINITION, GraphAgent_INSTRUCTION
from graph_fewshots import EXAMPLES

def process_dataset(dataset_names, base_dir, examples_dict, split, level_key='new_level'):
    data_list = []
    index = 0
    for name in dataset_names:
        target_search = os.path.join(base_dir, name)
        
        # Find all data.json files under this target
        found_files = []
        if os.path.exists(os.path.join(target_search, 'data.json')):
            found_files.append(os.path.join(target_search, 'data.json'))
        
        # Also check subdirectories
        subfiles = glob.glob(os.path.join(target_search, '*', 'data.json'))
        found_files.extend(subfiles)
        
        # Remove duplicates
        found_files = list(set(found_files))
        
        if not found_files:
            print(f"Warning: No data.json found for {name} in {base_dir}")
            continue

        for path in found_files:
            print(f"Processing {path}")
            rel_path = os.path.relpath(path, base_dir)
            graph_name = os.path.dirname(rel_path) # e.g. maple/Physics

            # Convert graph_name to domain format (maple/Biology -> maple_Biology)
            # This matches the keys in graph.domains config
            domain = graph_name.replace('/', '_')

            # graph_key (maple, dblp, etc.) for looking up examples and definitions
            parts = graph_name.split('/')
            graph_key = parts[0]

            ex_text = examples_dict.get(graph_key, "")
            def_text = GRAPH_DEFINITION.get(graph_key, "")
            
            with open(path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        question = item['question']
                        answer = item['answer']
                        try:
                            level_val = item[level_key]
                        except:
                            print(f"Error getting level for {item['qid']}")
                            level_val = None
                        
                        if level_val is None:
                            raise ValueError(f"Level is None for {item['qid']}")

                        prompt_text = GraphAgent_INSTRUCTION.format(
                            examples=ex_text,
                            graph_definition=def_text,
                            question=question,
                            scratchpad=""
                        )
                        
                        entry = {
                            'data_source': 'graph_cot',
                            'prompt': [{'role': 'user', 'content': prompt_text}],
                            'ability': 'graph',
                            'reward_model': {'ground_truth': str(answer), 'style': 'rule'},
                            'extra_info': {
                                'answer': str(answer),
                                'domain': domain,  # e.g. maple_Biology, dblp, biomedical
                                'graph_name': graph_name,  # kept for backward compat
                                'index': index,
                                'split': split,
                                'question': question,
                                'difficulty': level_val  # renamed from new_level for clarity
                            }
                        }
                        data_list.append(entry)
                        index += 1
                    except Exception as e:
                        print(f"Error processing line in {path}: {e}")
                        
    return data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_datasets", nargs="+", required=True)
    parser.add_argument("--test_datasets", nargs="+", required=True)
    parser.add_argument("--output_name", required=True, help="Folder name under GRBench")
    parser.add_argument(
        "--data_root",
        default=os.environ.get("GRAPHCOT_DATA_ROOT", "./data/processed_data"),
        help="Root directory containing Graph-CoT processed data (folders like dblp/, maple/*, biomedical/, ...).",
    )
    parser.add_argument("--level_key", default="new_level", help="The key to read level information from data items")
    args = parser.parse_args()

    # No longer need to load examples manually as they are imported
    examples_dict = EXAMPLES
    
    print("Generating Train Set...")
    train_data = process_dataset(args.train_datasets, args.data_root, examples_dict, "train", level_key=args.level_key)
    print(f"Train samples: {len(train_data)}")
    
    print("Generating Test Set...")
    test_data = process_dataset(args.test_datasets, args.data_root, examples_dict, "test", level_key=args.level_key)
    print(f"Test samples: {len(test_data)}")
    
    output_dir = os.path.join("./data", args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
