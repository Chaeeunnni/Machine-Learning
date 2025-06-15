import os
from models.kobert_encoder import KoBERTEncoder
from models.clip_encoder import CLIPEncoder
from models.matching import find_best_match
from utils.metadata_utils import load_json

import numpy as np

def join_text(context, utterance):
    return " ".join(context + [utterance])

if __name__ == "__main__":
    text_data = load_json("data/converted_dialogues.json")
    image_meta = load_json("data/image_metadata.json")

    kobert = KoBERTEncoder()
    clip_encoder = CLIPEncoder()

    print("[*] 이미지 임베딩 중...")
    image_embs, image_names = clip_encoder.encode_all_images(image_meta, "data/images")

    for sample in text_data[:5]:  # 예시 5개만 테스트
        input_text = join_text(sample["context"], sample["utterance"])
        text_emb = kobert.encode_text(input_text)
        idx, score = find_best_match(text_emb, image_embs)
        print(f"\n🧠 입력: {input_text}\n🎯 추천: {image_names[idx]} (유사도: {score:.2f})")
