import os
import warnings
from transformers import logging

# 경고 메시지 비활성화
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from models.hybrid_encoder import HybridEncoder
from models.matching import HybridMatcher
import glob
import numpy as np
import json


class HybridMemeRecommender:
    def __init__(self, metadata_file="data/enhanced_image_metadata.json"):
        print("[*] 하이브리드 짤 추천 시스템 초기화 중...")

        # 하이브리드 인코더와 매처 초기화
        self.hybrid_encoder = HybridEncoder()
        self.matcher = HybridMatcher(metadata_file)

        # 이미지 로딩 및 임베딩
        self.load_images_from_metadata(metadata_file)
        self.precompute_image_embeddings()

    def load_images_from_metadata(self, metadata_file):
        """메타데이터에서 이미지 로딩"""
        self.image_files = []

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                for item in metadata:
                    if item.get('processing_status') == 'success':
                        # processed_path 또는 filepath 사용
                        img_path = item.get('processed_path') or item.get('filepath')
                        if img_path and os.path.exists(img_path):
                            self.image_files.append(img_path)
                        else:
                            # 상대 경로로 시도
                            relative_path = f"data/images/{item.get('filepath', '')}"
                            if os.path.exists(relative_path):
                                self.image_files.append(relative_path)

                print(f"[*] 메타데이터에서 {len(self.image_files)}개 이미지 로드")

            except Exception as e:
                print(f"[!] 메타데이터 로드 실패: {e}")
                # 폴백: 기존 방식으로 이미지 로드
                self.load_images_fallback()
        else:
            print("[!] 메타데이터 파일이 없음. 폴백 모드로 로드")
            self.load_images_fallback()

    def load_images_fallback(self):
        """폴백: 디렉토리에서 직접 이미지 로드"""
        image_patterns = [
            "data/images/기쁨/**/*.jpg",
            "data/images/기쁨/**/*.png",
            "data/images/기쁨/**/*.jpeg",
            "data/images/기쁨/**/*.webp",
            "data/images/슬픔/**/*.jpg",
            "data/images/슬픔/**/*.png",
            "data/images/슬픔/**/*.jpeg",
            "data/images/슬픔/**/*.webp",
            "data/images/불안/**/*.jpg",
            "data/images/불안/**/*.png",
            "data/images/불안/**/*.jpeg",
            "data/images/불안/**/*.webp",
            "data/images/상처/**/*.jpg",
            "data/images/상처/**/*.png",
            "data/images/상처/**/*.jpeg",
            "data/images/상처/**/*.webp",
            "data/images/당황/**/*.jpg",
            "data/images/당황/**/*.png",
            "data/images/당황/**/*.jpeg",
            "data/images/당황/**/*.webp"
        ]

        self.image_files = []
        for pattern in image_patterns:
            found_files = glob.glob(pattern, recursive=True)
            self.image_files.extend(found_files)

        print(f"[*] 폴백 모드로 {len(self.image_files)}개 이미지 로드")

    def precompute_image_embeddings(self):
        """이미지 임베딩 미리 계산"""
        if not hasattr(self, 'image_files') or not self.image_files:
            print("[!] 이미지 파일이 없습니다. 이미지를 확인해주세요.")
            return

        print("[*] 이미지 임베딩 생성 중...")
        self.image_embeddings = []

        for i, img_path in enumerate(self.image_files):
            if i % 10 == 0:
                print(f"  진행률: {i}/{len(self.image_files)}")

            try:
                emb = self.hybrid_encoder.get_image_embedding(img_path)
                self.image_embeddings.append(emb)
            except Exception as e:
                print(f"[!] 이미지 임베딩 실패: {img_path}, 오류: {e}")
                # 0으로 채운 임베딩 추가 (에러 방지)
                self.image_embeddings.append(np.zeros(512))

        self.image_embeddings = np.array(self.image_embeddings)
        print(f"[*] 이미지 임베딩 생성 완료! (총 {len(self.image_embeddings)}개)")

    def recommend(self, dialogue_text):
        """하이브리드 방식으로 이미지 추천"""
        if not hasattr(self, 'image_embeddings') or len(self.image_embeddings) == 0:
            print("[!] 이미지 임베딩이 없습니다.")
            return None

        print(f"\n[*] 하이브리드 분석 중: {dialogue_text[:50]}...")

        # 하이브리드 매칭 수행
        best_idx, score, top_5, emotions, situations = self.matcher.find_best_match_hybrid(
            self.hybrid_encoder, dialogue_text, self.image_embeddings, self.image_files
        )

        return {
            'best_image': os.path.basename(self.image_files[best_idx]),
            'best_image_path': self.image_files[best_idx],
            'score': score,
            'emotions': emotions,
            'situations': situations,
            'top_5': top_5
        }

    def tune_weights(self, kobert_weight, clip_weight):
        """가중치 조정"""
        self.hybrid_encoder.set_weights(kobert_weight, clip_weight)


def main():
    # 하이브리드 추천 시스템 초기화
    recommender = HybridMemeRecommender()

    # 가중치 실험 (선택사항)
    print("\n[*] 가중치 실험 중...")
    recommender.tune_weights(0.7, 0.3)  # KoBERT 더 높은 가중치

    # 테스트 실행
    test_cases = [
        "일은 왜 해도 해도 끝이 없을까? 화가 난다.",
        "이번 달에 또 급여가 깎였어! 물가는 오르는데 월급만 자꾸 깎이니까 너무 화가 나.",
        "회사에 신입이 들어왔는데 말투가 거슬려. 스트레스 받아.",
        "오늘 정말 기분이 좋아! 승진 소식을 들었어!",
        "친구가 도움을 줘서 정말 감사해. 고마워!"
    ]

    print("\n" + "=" * 80)
    print("🎯 하이브리드(KoBERT + CLIP) 짤 추천 테스트!")
    print("=" * 80)

    for i, dialogue in enumerate(test_cases, 1):
        print(f"\n【하이브리드 테스트 {i}】")
        result = recommender.recommend(dialogue)

        if result is None:
            print("❌ 추천 실패")
            continue

        print(f"🧠 입력: {dialogue}")
        print(f"🎯 추천: {result['best_image']}")
        print(f"📊 최종 점수: {result['score']:.4f}")
        print(f"😊 감지 감정: {result['emotions']}")
        print(f"🏢 감지 상황: {result['situations']}")
        print(f"🏆 Top 5:")
        for j, item in enumerate(result['top_5'], 1):
            print(f"   {j}. {item['filename']} - {item['category']}/{item['subcategory']} ({item['score']:.4f})")


if __name__ == "__main__":
    main()