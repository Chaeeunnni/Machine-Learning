import os
import warnings
from transformers import logging

# 경고 메시지 비활성화
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from models.clip_encoder import CLIPEncoder
from models.matching import EnhancedMatcher
from utils.data_loader import DialogueDataLoader
from utils.evaluator import MemeRecommendationEvaluator
import glob
import numpy as np
import json


class TrainableRecommendationSystem:
    def __init__(self, dialogue_metadata="data/converted_dialogues.json",
                 image_metadata="data/image_metadata.json"):
        print("[*] 학습 가능한 추천 시스템 초기화...")

        # 데이터 로더 초기화
        self.data_loader = DialogueDataLoader(dialogue_metadata, image_metadata)

        # 모델 초기화
        self.clip_encoder = CLIPEncoder()
        self.matcher = EnhancedMatcher(image_metadata)

        # 이미지 파일들 로드
        self.load_images_from_metadata(image_metadata)

        # 이미지 임베딩 미리 계산
        self.precompute_image_embeddings()

    def load_images_from_metadata(self, metadata_file):
        """메타데이터에서 이미지 로드"""
        self.image_files = []

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            for item in metadata:
                if item.get('processing_status') == 'success':
                    img_path = item.get('processed_path') or item.get('filepath')
                    if img_path and os.path.exists(img_path):
                        self.image_files.append(img_path)

        print(f"[*] 총 {len(self.image_files)}개 이미지 로드")

    def precompute_image_embeddings(self):
        """이미지 임베딩 미리 계산"""
        print("[*] 이미지 임베딩 생성 중...")
        self.image_embeddings = []

        for i, img_path in enumerate(self.image_files):
            if i % 10 == 0:
                print(f"  진행률: {i}/{len(self.image_files)}")

            emb = self.clip_encoder.encode_image(img_path)
            self.image_embeddings.append(emb)

        self.image_embeddings = np.array(self.image_embeddings)
        print("[*] 이미지 임베딩 생성 완료!")

    def recommend(self, dialogue_text):
        """대화 텍스트에 대한 이미지 추천"""
        text_emb = self.clip_encoder.get_text_embedding(dialogue_text)

        best_idx, score, top_5, emotions, situations = self.matcher.find_best_match(
            text_emb, self.image_embeddings, self.image_files, dialogue_text
        )

        return {
            'best_image': os.path.basename(self.image_files[best_idx]),
            'best_image_path': self.image_files[best_idx],
            'score': score,
            'emotions': emotions,
            'situations': situations,
            'top_5': top_5
        }

    def train_and_evaluate(self, test_size=0.2):
        """학습 데이터로 성능 평가"""
        print("\n[*] 데이터 분할 중...")
        train_data, test_data = self.data_loader.split_data(test_size=test_size)

        print("\n[*] 성능 평가 시작...")
        evaluator = MemeRecommendationEvaluator(self, test_data)
        results = evaluator.evaluate_recommendations()

        # 결과 출력
        evaluator.print_evaluation_report(results)

        # 결과 저장
        evaluator.save_evaluation_results(results)

        return results


def main():
    # 시스템 초기화 및 평가
    system = TrainableRecommendationSystem()

    # 학습 및 평가 실행
    results = system.train_and_evaluate(test_size=0.2)

    # 추가 테스트 케이스로 실시간 테스트
    print("\n" + "=" * 60)
    print("🎯 실시간 테스트")
    print("=" * 60)

    test_cases = [
        "일은 왜 해도 해도 끝이 없을까? 화가 난다.",
        "오늘 정말 기분이 좋아! 승진했어!",
        "연인과 헤어져서 너무 슬퍼. 어떻게 해야 할지 모르겠어.",
        "회사 상사가 너무 짜증나. 그만두고 싶어."
    ]

    for i, dialogue in enumerate(test_cases, 1):
        print(f"\n【실시간 테스트 {i}】")
        result = system.recommend(dialogue)

        print(f"🧠 입력: {dialogue}")
        print(f"🎯 추천: {result['best_image']}")
        print(f"📊 유사도: {result['score']:.4f}")
        print(f"😊 감지 감정: {result['emotions']}")
        print(f"🏢 감지 상황: {result['situations']}")


if __name__ == "__main__":
    main()