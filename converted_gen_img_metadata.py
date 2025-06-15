import json

def add_tags_to_image_metadata(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        category = item.get("category", "")
        subcategory = item.get("subcategory", "")
        item["tags"] = [category, subcategory]

    save_path = output_path if output_path else input_path
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ 태그가 추가된 메타데이터가 '{save_path}'에 저장되었습니다.")

# ✅ 사용 예시
add_tags_to_image_metadata("data/image_metadata.json")
