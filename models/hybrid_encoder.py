import torch
import numpy as np
from .kobert_encoder import KoBERTEncoder
from .clip_encoder import CLIPEncoder
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class HybridEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("[*] 하이브리드 인코더 초기화 중...")

        # 두 인코더 초기화
        self.kobert_encoder = KoBERTEncoder(device)
        self.clip_encoder = CLIPEncoder(device)

        # 차원 통일을 위한 PCA (선택사항)
        self.use_pca = True
        self.target_dim = 512  # CLIP 차원에 맞춤
        self.kobert_pca = None

        # 가중치 설정 (조정 가능)
        self.kobert_weight = 0.6  # KoBERT가 한국어에 특화되어 있으므로 더 높은 가중치
        self.clip_weight = 0.4  # CLIP은 멀티모달 특성 활용

        print(f"[*] 하이브리드 인코더 초기화 완료 (KoBERT: {self.kobert_weight}, CLIP: {self.clip_weight})")

    def _align_dimensions(self, kobert_emb, clip_emb):
        """차원을 맞춰주는 함수 - 안정적인 방법"""
        try:
            # KoBERT 임베딩 차원 확인
            kobert_dim = len(kobert_emb)
            clip_dim = len(clip_emb)

            print(f"[DEBUG] KoBERT 차원: {kobert_dim}, CLIP 차원: {clip_dim}")

            # 목표 차원 (CLIP에 맞춤)
            target_dim = 512

            # KoBERT 차원 조정
            if kobert_dim > target_dim:
                # 768 -> 512: 간격을 두고 샘플링
                indices = np.linspace(0, kobert_dim - 1, target_dim, dtype=int)
                kobert_aligned = kobert_emb[indices]
            elif kobert_dim < target_dim:
                # 256 -> 512: 제로 패딩
                padding = target_dim - kobert_dim
                kobert_aligned = np.pad(kobert_emb, (0, padding), mode='constant')
            else:
                # 같은 차원
                kobert_aligned = kobert_emb

            # CLIP 차원 조정
            if clip_dim != target_dim:
                if clip_dim > target_dim:
                    clip_aligned = clip_emb[:target_dim]
                else:
                    padding = target_dim - clip_dim
                    clip_aligned = np.pad(clip_emb, (0, padding), mode='constant')
            else:
                clip_aligned = clip_emb

            print(f"[DEBUG] 정렬 후 - KoBERT: {len(kobert_aligned)}, CLIP: {len(clip_aligned)}")

            return kobert_aligned, clip_aligned

        except Exception as e:
            print(f"[!] 차원 정렬 실패: {e}")
            # 안전한 폴백: 둘 다 512차원으로 맞춤
            target_dim = 512

            # KoBERT 처리
            if len(kobert_emb) >= target_dim:
                kobert_safe = kobert_emb[:target_dim]
            else:
                kobert_safe = np.pad(kobert_emb, (0, target_dim - len(kobert_emb)), mode='constant')

            # CLIP 처리
            if len(clip_emb) >= target_dim:
                clip_safe = clip_emb[:target_dim]
            else:
                clip_safe = np.pad(clip_emb, (0, target_dim - len(clip_emb)), mode='constant')

            return kobert_safe, clip_safe

    def get_text_embedding(self, text):
        """KoBERT + CLIP 텍스트 임베딩 결합"""
        try:
            # 1. 각각의 임베딩 생성
            kobert_emb = self.kobert_encoder.encode_text(text)
            clip_emb = self.clip_encoder.get_text_embedding(text)

            # 2. 차원 맞추기
            kobert_aligned, clip_aligned = self._align_dimensions(kobert_emb, clip_emb)

            # 3. 정규화
            kobert_aligned = normalize([kobert_aligned], norm='l2')[0]
            clip_aligned = normalize([clip_aligned], norm='l2')[0]

            # 4. 가중 평균으로 결합
            combined_emb = (self.kobert_weight * kobert_aligned +
                            self.clip_weight * clip_aligned)

            # 5. 다시 정규화
            combined_emb = normalize([combined_emb], norm='l2')[0]

            return combined_emb

        except Exception as e:
            print(f"[!] 텍스트 임베딩 생성 실패: {text[:50]}..., 오류: {e}")
            # 폴백: CLIP만 사용
            return self.clip_encoder.get_text_embedding(text)

    def get_text_embedding_concat(self, text):
        """연결(concatenation) 방식 - 대안"""
        try:
            kobert_emb = self.kobert_encoder.encode_text(text)
            clip_emb = self.clip_encoder.get_text_embedding(text)

            # 정규화 후 연결
            kobert_emb = normalize([kobert_emb], norm='l2')[0]
            clip_emb = normalize([clip_emb], norm='l2')[0]

            # 연결: [768차원 + 512차원] = 1280차원
            combined_emb = np.concatenate([kobert_emb, clip_emb])

            return combined_emb

        except Exception as e:
            print(f"[!] 연결 방식 임베딩 생성 실패: {text[:50]}..., 오류: {e}")
            # 폴백: CLIP만 사용
            return self.clip_encoder.get_text_embedding(text)

    def get_dual_similarities(self, text, image_embs):
        """두 모델 각각의 유사도 계산 후 결합"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # KoBERT 텍스트 임베딩
            kobert_text_emb = self.kobert_encoder.encode_text(text)

            # CLIP 텍스트 임베딩
            clip_text_emb = self.clip_encoder.get_text_embedding(text)

            # 이미지 임베딩은 CLIP 차원(512)이므로 KoBERT도 맞춰줌
            kobert_aligned, _ = self._align_dimensions(kobert_text_emb, clip_text_emb)

            # 정규화
            kobert_aligned = normalize([kobert_aligned], norm='l2')[0]
            clip_text_emb = normalize([clip_text_emb], norm='l2')[0]
            image_embs_norm = normalize(image_embs, norm='l2')

            # 각각의 유사도 계산
            kobert_similarities = cosine_similarity([kobert_aligned], image_embs_norm)[0]
            clip_similarities = cosine_similarity([clip_text_emb], image_embs_norm)[0]

            # 가중 합산
            combined_similarities = (self.kobert_weight * kobert_similarities +
                                     self.clip_weight * clip_similarities)

            return combined_similarities, kobert_similarities, clip_similarities

        except Exception as e:
            print(f"[!] 듀얼 유사도 계산 실패: {text[:50]}..., 오류: {e}")
            # 폴백: CLIP만 사용
            from sklearn.metrics.pairwise import cosine_similarity
            clip_text_emb = self.clip_encoder.get_text_embedding(text)
            clip_text_emb = normalize([clip_text_emb], norm='l2')[0]
            image_embs_norm = normalize(image_embs, norm='l2')
            clip_similarities = cosine_similarity([clip_text_emb], image_embs_norm)[0]
            return clip_similarities, np.zeros_like(clip_similarities), clip_similarities

    def get_image_embedding(self, image_path):
        """이미지 임베딩은 CLIP만 사용"""
        try:
            return self.clip_encoder.get_image_embedding(image_path)
        except Exception as e:
            print(f"[!] 이미지 임베딩 실패: {image_path}, 오류: {e}")
            return np.zeros(512)  # CLIP 기본 차원

    def set_weights(self, kobert_weight, clip_weight):
        """가중치 동적 조정"""
        total = kobert_weight + clip_weight
        if total == 0:
            print("[!] 가중치 합이 0입니다. 기본값을 사용합니다.")
            return

        self.kobert_weight = kobert_weight / total
        self.clip_weight = clip_weight / total
        print(f"[*] 가중치 업데이트: KoBERT={self.kobert_weight:.2f}, CLIP={self.clip_weight:.2f}")

    def set_dimension_method(self, use_pca=True, target_dim=512):
        """차원 정렬 방법 설정"""
        self.use_pca = use_pca
        self.target_dim = target_dim
        self.kobert_pca = None  # 재설정
        print(f"[*] 차원 정렬 방법: {'PCA' if use_pca else '단순 축소'}, 목표 차원: {target_dim}")

    def get_kobert_embedding(self, text):
        """KoBERT 임베딩만 반환 (디버깅용)"""
        return self.kobert_encoder.encode_text(text)

    def get_clip_text_embedding(self, text):
        """CLIP 텍스트 임베딩만 반환 (디버깅용)"""
        return self.clip_encoder.get_text_embedding(text)

    def get_embedding_info(self, text):
        """임베딩 정보 반환 (디버깅용)"""
        kobert_emb = self.kobert_encoder.encode_text(text)
        clip_emb = self.clip_encoder.get_text_embedding(text)
        combined_emb = self.get_text_embedding(text)

        return {
            'kobert_shape': kobert_emb.shape,
            'clip_shape': clip_emb.shape,
            'combined_shape': combined_emb.shape,
            'kobert_norm': np.linalg.norm(kobert_emb),
            'clip_norm': np.linalg.norm(clip_emb),
            'combined_norm': np.linalg.norm(combined_emb)
        }

    def test_compatibility(self):
        """호환성 테스트"""
        try:
            test_text = "테스트 문장입니다."
            print(f"[*] 호환성 테스트 시작: '{test_text}'")

            # 각 임베딩 생성 테스트
            kobert_emb = self.kobert_encoder.encode_text(test_text)
            print(f"  KoBERT 임베딩: {kobert_emb.shape}")

            clip_emb = self.clip_encoder.get_text_embedding(test_text)
            print(f"  CLIP 임베딩: {clip_emb.shape}")

            combined_emb = self.get_text_embedding(test_text)
            print(f"  결합 임베딩: {combined_emb.shape}")

            concat_emb = self.get_text_embedding_concat(test_text)
            print(f"  연결 임베딩: {concat_emb.shape}")

            print("[✓] 호환성 테스트 성공!")
            return True

        except Exception as e:
            print(f"[✗] 호환성 테스트 실패: {e}")
            return False


# 테스트용 메인 함수
if __name__ == "__main__":
    # 하이브리드 인코더 테스트
    encoder = HybridEncoder()

    # 호환성 테스트
    if encoder.test_compatibility():
        # 실제 텍스트로 테스트
        test_texts = [
            "오늘 기분이 좋아!",
            "일이 너무 많아서 스트레스 받아.",
            "친구가 도움을 줘서 감사해."
        ]

        for text in test_texts:
            info = encoder.get_embedding_info(text)
            print(f"\n텍스트: {text}")
            print(f"임베딩 정보: {info}")