import os
import json
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration (anonymous & portable):
# - By default we assume data lives under this repo's `data/` directory.
# - You can override everything via environment variables.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKDIR = os.environ.get("GRAPHDANCER_WORKDIR", REPO_ROOT)

PROCESSED_DATA_DIR = os.environ.get(
    "PROCESSED_DATA_DIR",
    os.path.join(WORKDIR, "data", "processed_data_oldTemp_withError"),
)
LABELED_DATA_DIR = os.environ.get(
    "LABELED_DATA_DIR",
    os.path.join(WORKDIR, "data", "difficulty_level_labeled_data"),
)
TEMPLATE_FILE = os.environ.get(
    "TEMPLATE_FILE",
    os.path.join(REPO_ROOT, "scripts", "curriculum", "graphcot_templates.json"),
)
AWARE_TEMPLATE_FILE = os.environ.get(
    "AWARE_TEMPLATE_FILE",
    os.path.join(REPO_ROOT, "scripts", "curriculum", "graph_aware_templates_v0.json"),
)

# Mappings
# processed_dir_name -> labeled_filename
DIR_TO_LABEL_FILE = {
    "maple/Biology": "biology.json",
    "maple/Chemistry": "chemistry.json",
    "maple/Materials_Science": "materials_science.json",
    "maple/Medicine": "medicine.json",
    "maple/Physics": "physics.json",
    "dblp": "computer_science.json",
    "amazon": "amazon.json",
    "biomedical": "healthcare.json",
    "goodreads": "literature.json",
    "legal": "legal.json"
}

# processed_dir_name -> graph_type (for templates)
DIR_TO_GRAPH_TYPE = {
    "maple/Biology": "Academic Graphs",
    "maple/Chemistry": "Academic Graphs",
    "maple/Materials_Science": "Academic Graphs",
    "maple/Medicine": "Academic Graphs",
    "maple/Physics": "Academic Graphs",
    "dblp": "Academic Graphs",
    "amazon": "E-commerce Graph",
    "biomedical": "Healthcare Graph",
    "goodreads": "Literature Graph",
    "legal": "Legal Graph"
}

client = openai.OpenAI()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_template_match(question, templates):
    if not templates:
        return None
    
    # Format prompt
    prompt = f"Question: \"{question}\"\n\nSelect the template that best matches the structure of the question above from the following list:\n"
    for i, t in enumerate(templates):
        prompt += f"{i+1}. {t}\n"
    prompt += "\nReturn ONLY the exact text of the matching template, nothing else."

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that matches questions to templates."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up content (sometimes returns quotes)
        content_clean = content.strip('"').strip("'")
        
        # Try exact match
        if content_clean in templates:
            return content_clean
        
        # Try match if OpenAI returns "Template: ..." or similar
        for t in templates:
            if t in content_clean:
                return t
        
        # Fallback: return clean content, maybe it matches later or is just slightly off
        return content_clean
    except Exception as e:
        print(f"Error matching template for question '{question[:50]}...': {e}")
        return None

def process_file(rel_path, label_file, graph_type, graphcot_templates, aware_templates_map):
    data_path = os.path.join(PROCESSED_DATA_DIR, rel_path, "data.json")
    label_path = os.path.join(LABELED_DATA_DIR, label_file)
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    print(f"Processing {rel_path} using labels from {label_file}...")

    # Read data
    data_items = load_jsonl(data_path)
    
    # Read labels
    label_items = load_jsonl(label_path)
    label_map = {str(item['qid']): item.get('level') for item in label_items}
    
    updated_data = []
    futures = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    # Adjust max_workers based on rate limits
    with ThreadPoolExecutor(max_workers=10) as executor:
        for item in data_items:
            qid = str(item['qid'])
            level_raw = label_map.get(qid)
            
            if not level_raw:
                print(f"Warning: QID {qid} not found in labels for {rel_path}")
                item['level'] = None
                # We can't find template if we don't know level (needed to look up template list)
                updated_data.append(item)
                continue
                
            # (1) Add level
            level_lower = level_raw.lower()
            item['level'] = level_lower
            
            # Get templates for this graph type and level
            level_cap = level_lower.capitalize()
            
            # Check if level exists in graphcot templates
            # graphcot uses "Easy", "Medium", "Hard". 
            # labeled data might use "easy", "medium", "hard"
            
            target_templates = graphcot_templates.get(graph_type, {}).get(level_cap, [])
            # -> 拼接所有难度的templates
            # graph_type_templates = graphcot_templates.get(graph_type, {})
            # all_templates = []
            # for diff_list in graph_type_templates.values():
            #     all_templates.extend(diff_list)
            # target_templates = all_templates
            
            if not target_templates:
                # Could happen if level is "OOD" or something not in graphcot
                # User said graphcot has templates for each level.
                # If labeled data has levels not in graphcot, we skip template matching.
                # print(f"No templates found for {graph_type} - {level_cap}")
                raise ValueError(f"No templates found for {graph_type} - {level_cap}")
                item['template'] = None
                item['new_level'] = None
                updated_data.append(item)
                continue

            # (2) Queue template matching
            future = executor.submit(get_template_match, item['question'], target_templates)
            futures[future] = item

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Matching templates for {rel_path}"):
            item = futures[future]
            try:
                matched_template = future.result()
                
                # (2) Add template
                item['template'] = matched_template
                
                # (3) Add new_level
                new_level = None
                if matched_template:
                    # Look up in graph_aware_templates
                    aware_levels = aware_templates_map.get(graph_type, {})
                    found = False
                    for lvl, tmpls in aware_levels.items():
                        if matched_template in tmpls:
                            new_level = lvl
                            found = True
                            break
                    
                    if not found:
                        # Try relaxed matching?
                        pass
                
                item['new_level'] = new_level
                
            except Exception as e:
                print(f"Error in future processing: {e}")
                item['template'] = None
                item['new_level'] = None
            
            updated_data.append(item)

    # Sort by QID to maintain order (assuming numeric qid)
    def get_sort_key(x):
        try:
            return int(x['qid'])
        except ValueError:
            return x['qid']
            
    updated_data.sort(key=get_sort_key)
    
    # Save back to data.json
    save_jsonl(updated_data, data_path)
    print(f"Finished updating {data_path}\n")

def main():
    # Load templates
    if not os.path.exists(TEMPLATE_FILE):
        print(f"Template file not found: {TEMPLATE_FILE}")
        return
    if not os.path.exists(AWARE_TEMPLATE_FILE):
        print(f"Aware template file not found: {AWARE_TEMPLATE_FILE}")
        return

    with open(TEMPLATE_FILE, 'r') as f:
        graphcot_templates = json.load(f)
    
    with open(AWARE_TEMPLATE_FILE, 'r') as f:
        aware_templates = json.load(f)

    # Iterate over directories
    for rel_path, label_file in DIR_TO_LABEL_FILE.items():
        graph_type = DIR_TO_GRAPH_TYPE.get(rel_path)
        if not graph_type:
            continue
            
        process_file(rel_path, label_file, graph_type, graphcot_templates, aware_templates)

if __name__ == "__main__":
    main()

