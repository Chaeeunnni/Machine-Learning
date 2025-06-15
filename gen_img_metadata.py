import os
import json

def generate_image_metadata(base_dir, output_path="image_metadata.json"):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    metadata = []

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)
            if not os.path.isdir(subcategory_path):
                continue

            for filename in os.listdir(subcategory_path):
                if filename.lower().endswith(image_extensions):
                    entry = {
                        "filename": filename,
                        "filepath": os.path.join(category, subcategory, filename).replace("\\", "/"),
                        "subcategory": subcategory,
                        "category": category
                    }
                    metadata.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 이미지 메타데이터 {len(metadata)}개가 {output_path}에 저장되었습니다.")

# ✅ 사용 예시
generate_image_metadata("D:/2025-1(4-2)/MLproject/data/images")
