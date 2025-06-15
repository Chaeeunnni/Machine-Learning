import os
import warnings
from transformers import logging

# 경고 메시지 비활성화
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from models.clip_encoder import CLIPEncoder
from models.matching import EnhancedMatcher
import glob
import numpy as np
import json


class ImprovedMemeRecommender:
    def __init__(self, metadata_file="data/enhanced_image_metadata.json"):
        print("[*] 짤 추천 시스템 초기화 중...")

        # 모델 초기화
        self.clip_encoder = CLIPEncoder()
        self.matcher = EnhancedMatcher(metadata_file)

        # 메타데이터에서 이미지 파일 경로 추출
        self.load_images_from_metadata(metadata_file)

        print(f"[*] 총 {len(self.image_files)}개 이미지 발견")

        # 이미지 임베딩 미리 계산
        print("[*] 이미지 임베딩 생성 중...")
        self.image_embeddings = []
        for i, img_path in enumerate(self.image_files):
            if i % 10 == 0:
                print(f"  진행률: {i}/{len(self.image_files)}")

            emb = self.clip_encoder.encode_image(img_path)
            self.image_embeddings.append(emb)

        self.image_embeddings = np.array(self.image_embeddings)
        print("[*] 이미지 임베딩 생성 완료!")

    def load_images_from_metadata(self, metadata_file):
        """메타데이터 파일에서 이미지 경로 로드"""
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
            "data/images/슬픔/**/*.jpg",
            "data/images/슬픔/**/*.png",
            "data/images/불안/**/*.jpg",
            "data/images/불안/**/*.png",
            "data/images/상처/**/*.jpg",
            "data/images/상처/**/*.png",
            "data/images/당황/**/*.jpg",
            "data/images/당황/**/*.png"
        ]

        self.image_files = []
        for pattern in image_patterns:
            self.image_files.extend(glob.glob(pattern, recursive=True))

    def recommend(self, dialogue_text):
        """대화 텍스트에 대한 이미지 추천"""
        print(f"\n[*] 분석 중: {dialogue_text[:50]}...")

        # 텍스트 임베딩 생성
        text_emb = self.clip_encoder.get_text_embedding(dialogue_text)

        # 매칭 수행
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


def main():
    # 추천 시스템 초기화
    recommender = ImprovedMemeRecommender()

    # 테스트 대화들
    test_cases = [
        "일은 왜 해도 해도 끝이 없을까? 화가 난다.",
        "이번 달에 또 급여가 깎였어! 물가는 오르는데 월급만 자꾸 깎이니까 너무 화가 나.",
        "회사에 신입이 들어왔는데 말투가 거슬려. 스트레스 받아.",
        "직장에서 막내라는 이유로 나에게만 온갖 심부름을 시켜. 정말 분하고 섭섭해.",
        "오늘 정말 기분이 좋아! 승진 소식을 들었어!",
        "연인과 헤어져서 너무 슬퍼. 어떻게 해야 할지 모르겠어.",
        "친구가 도움을 줘서 정말 감사해. 고마워!",
        "이번 프로젝트가 성공해서 너무 기뻐!"
    ]

    print("\n" + "=" * 70)
    print("🎯 향상된 짤 추천 테스트 시작!")
    print("=" * 70)

    for i, dialogue in enumerate(test_cases, 1):
        print(f"\n【테스트 {i}】")
        result = recommender.recommend(dialogue)

        print(f"🧠 입력: {dialogue}")
        print(f"🎯 추천: {result['best_image']}")
        print(f"📊 유사도: {result['score']:.4f}")
        print(f"😊 감지 감정: {result['emotions']}")
        print(f"🏢 감지 상황: {result['situations']}")
        print(f"🏆 Top 5:")
        for j, item in enumerate(result['top_5'], 1):
            print(f"   {j}. {item['filename']} - {item['category']}/{item['subcategory']} ({item['score']:.4f})")


if __name__ == "__main__":
    main()