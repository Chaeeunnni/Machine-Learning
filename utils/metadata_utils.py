import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_tags_to_metadata(input_path):
    data = load_json(input_path)
    for item in data:
        category = item.get("category")
        sub = item.get("subcategory")
        item["tags"] = [category, sub]
    save_json(data, input_path)