from PIL import Image
import torch
import clip
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CLIPEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.max_length = 77  # CLIP의 최대 토큰 길이

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.cpu().numpy()[0]

    def truncate_korean_text(self, text, max_tokens=70):
        """한국어 텍스트를 CLIP 토큰 길이에 맞게 자르기"""
        # 기본 정리
        text = text.strip()

        # 빈 텍스트 처리
        if not text:
            return "기본 텍스트"

        # 대화 형식에서 중요한 부분만 추출
        if 'A:' in text and 'B:' in text:
            # 마지막 발화만 사용 (가장 최근 감정 상태)
            parts = re.split(r'[AB]:', text)
            if len(parts) > 1:
                text = parts[-1].strip()

        # 문장 단위로 자르기
        sentences = re.split(r'[.!?]', text)
        truncated = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            test_text = truncated + sentence.strip() + "."
            # 대략적인 토큰 수 추정 (한국어는 보통 1.5-2자당 1토큰)
            estimated_tokens = len(test_text) // 1.5

            if estimated_tokens <= max_tokens:
                truncated = test_text
            else:
                break

        # 빈 결과 방지
        if not truncated.strip():
            # 최소한의 텍스트 보장 (글자 수 기준으로 자르기)
            char_limit = int(max_tokens * 1.5)  # 토큰당 1.5자 추정
            if len(text) > char_limit:
                truncated = text[:char_limit] + "..."
            else:
                truncated = text

        return truncated.strip() if truncated.strip() else "기본 텍스트"

    def get_text_embedding(self, text):
        """텍스트를 임베딩으로 변환 - 오류 처리 강화"""
        try:
            # 입력 검증
            if not text or not isinstance(text, str):
                print(f"잘못된 텍스트 입력: {text}")
                text = "기본 텍스트"

            # 텍스트 전처리 및 길이 제한
            processed_text = self.truncate_korean_text(text)

            # CLIP 토큰화 시도 (truncate=True 옵션 사용)
            text_tokens = clip.tokenize([processed_text], truncate=True).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # 정규화
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            result = text_features.cpu().numpy()[0]
            if not isinstance(result, np.ndarray):
                result = np.array(result)

            if result.ndim == 0:
                result = result.reshape(1)

            return result.astype(np.float32)

        except Exception as e:
            print(f"텍스트 임베딩 생성 실패: {text}")
            print(f"오류: {e}")

            # 폴백 1: 매우 짧은 텍스트로 재시도
            try:
                fallback_text = text[:20] if len(text) > 20 else text
                if not fallback_text.strip():
                    fallback_text = "기본"

                text_tokens = clip.tokenize([fallback_text], truncate=True).to(self.device)

                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                print(f"폴백 성공: {fallback_text}")
                return text_features.cpu().numpy()[0]

            except Exception as e2:
                print(f"폴백도 실패: {e2}")

                # 폴백 2: 영어 기본 텍스트 사용
                try:
                    default_text = "default text"
                    text_tokens = clip.tokenize([default_text], truncate=True).to(self.device)

                    with torch.no_grad():
                        text_features = self.model.encode_text(text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    print(f"영어 기본 텍스트로 폴백 성공")
                    return text_features.cpu().numpy()[0]

                except Exception as e3:
                    print(f"모든 폴백 실패: {e3}")
                    # 최후의 수단: 영벡터 반환 (CLIP 임베딩 차원은 512)
                    return np.zeros(512, dtype=np.float32)

    def get_dual_similarities(self, text_emb, image_embeddings, emotions=None, situations=None):
        """
        듀얼 유사도 계산 - 텍스트 임베딩과 감정/상황 가중치를 결합

        Args:
            text_emb: 텍스트 임베딩 벡터
            image_embeddings: 이미지 임베딩 배열
            emotions: 감지된 감정 리스트
            situations: 감지된 상황 리스트

        Returns:
            combined_similarities: 결합된 유사도 점수
            text_similarities: 순수 텍스트 유사도
            weighted_similarities: 가중치 적용된 유사도
        """
        try:
            # 1. 입력 타입 검증
            if isinstance(text_emb, str):
                print(f"[!] get_dual_similarities: 텍스트 임베딩이 문자열입니다. 다시 임베딩 생성...")
                text_emb = self.get_text_embedding(text_emb)

            if text_emb is None:
                print(f"[!] get_dual_similarities: 텍스트 임베딩이 None입니다.")
                return np.zeros(len(image_embeddings)), np.zeros(len(image_embeddings)), np.zeros(len(image_embeddings))

            # numpy 배열 확인
            if not isinstance(text_emb, np.ndarray):
                text_emb = np.array(text_emb)


            # 1. 순수 텍스트 유사도 계산
            text_similarities = self._calculate_text_similarity(text_emb, image_embeddings)

            # 2. 감정/상황 가중치 계산
            emotion_weights = self._calculate_emotion_weights(emotions) if emotions else 1.0
            situation_weights = self._calculate_situation_weights(situations) if situations else 1.0

            # 3. 가중치 적용된 유사도
            weighted_similarities = text_similarities * emotion_weights * situation_weights

            # 4. 결합된 최종 유사도 (텍스트 70% + 가중치 30%)
            combined_similarities = (0.7 * text_similarities + 0.3 * weighted_similarities)

            return combined_similarities, text_similarities, weighted_similarities

        except Exception as e:
            print(f"[!] 듀얼 유사도 계산 실패: {e}")
            # 폴백: 기본 텍스트 유사도만 반환
            text_similarities = self._calculate_text_similarity(text_emb, image_embeddings)
            return text_similarities, text_similarities, text_similarities

    def _calculate_text_similarity(self, text_emb, image_embeddings):
        """순수 텍스트-이미지 유사도 계산"""
        try:
            # 1. 입력 타입 검증 및 변환
            if text_emb is None:
                print("[!] 텍스트 임베딩이 None입니다")
                return np.zeros(len(image_embeddings))

            # 문자열이나 잘못된 타입이 들어온 경우 처리
            if isinstance(text_emb, str):
                print(f"[!] 텍스트 임베딩이 문자열입니다: {text_emb[:50]}...")
                print("[!] 문자열을 다시 임베딩으로 변환합니다")
                text_emb = self.get_text_embedding(text_emb)
                if text_emb is None:
                    return np.zeros(len(image_embeddings))

            # numpy 배열로 변환
            if not isinstance(text_emb, np.ndarray):
                try:
                    text_emb = np.array(text_emb)
                except:
                    print(f"[!] 텍스트 임베딩을 numpy 배열로 변환할 수 없습니다: {type(text_emb)}")
                    return np.zeros(len(image_embeddings))

            # 2. 차원 검증
            if text_emb.size == 0:
                print("[!] 빈 텍스트 임베딩")
                return np.zeros(len(image_embeddings))

            if len(image_embeddings) == 0:
                print("[!] 빈 이미지 임베딩")
                return np.array([])

            # 3. 차원 조정
            if text_emb.ndim == 1:
                text_emb = text_emb.reshape(1, -1)  # (1, 512)
            elif text_emb.ndim > 2:
                print(f"[!] 텍스트 임베딩 차원이 너무 큽니다: {text_emb.shape}")
                text_emb = text_emb.reshape(1, -1)

            # 4. 이미지 임베딩도 검증
            if not isinstance(image_embeddings, np.ndarray):
                try:
                    image_embeddings = np.array(image_embeddings)
                except:
                    print(f"[!] 이미지 임베딩을 numpy 배열로 변환할 수 없습니다: {type(image_embeddings)}")
                    return np.zeros(text_emb.shape[0])

            # 5. 차원 호환성 검사
            if text_emb.shape[1] != image_embeddings.shape[1]:
                print(f"[!] 임베딩 차원 불일치: 텍스트 {text_emb.shape}, 이미지 {image_embeddings.shape}")
                # 차원 맞추기 시도
                min_dim = min(text_emb.shape[1], image_embeddings.shape[1])
                text_emb = text_emb[:, :min_dim]
                image_embeddings = image_embeddings[:, :min_dim]

            # 6. 코사인 유사도 계산
            similarities = cosine_similarity(text_emb, image_embeddings)[0]  # (n_images,)

            # 7. NaN 및 무한대 값 처리
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

            return similarities

        except Exception as e:
            print(f"[!] 텍스트 유사도 계산 실패: {e}")
            print(f"[!] 텍스트 임베딩 타입: {type(text_emb)}")
            if hasattr(text_emb, 'shape'):
                print(f"[!] 텍스트 임베딩 shape: {text_emb.shape}")
            print(f"[!] 이미지 임베딩 타입: {type(image_embeddings)}")
            if hasattr(image_embeddings, 'shape'):
                print(f"[!] 이미지 임베딩 shape: {image_embeddings.shape}")

            # 안전한 폴백
            if hasattr(image_embeddings, '__len__'):
                return np.zeros(len(image_embeddings))
            else:
                return np.zeros(1)

    def _calculate_emotion_weights(self, emotions):
        """감정 기반 가중치 계산"""
        if not emotions:
            return 1.0

        # 감정별 가중치 매핑 (더 정확한 매핑)
        emotion_weights = {
            '기쁘다': 1.2,
            '슬프다': 1.1,
            '화나다': 1.15,
            '무섭다': 1.1,
            '놀랍다': 1.05,
            '감사하다': 1.1,
            '스트레스': 1.2,
            '걱정': 1.15,
            '행복': 1.2,
            '우울': 1.1,
            '부담': 1.15,
            '짜증': 1.15,
            '불안': 1.1
        }

        # 감지된 감정들의 평균 가중치
        weights = [emotion_weights.get(emotion, 1.0) for emotion in emotions]
        return np.mean(weights) if weights else 1.0

    def _calculate_situation_weights(self, situations):
        """상황 기반 가중치 계산"""
        if not situations:
            return 1.0

        # 상황별 가중치 매핑 (더 정확한 매핑)
        situation_weights = {
            '직장': 1.15,
            '가족': 1.1,
            '연애': 1.1,
            '친구': 1.05,
            '학교': 1.1,
            '돈': 1.2,
            '건강': 1.15,
            '취업': 1.2,
            '축하': 1.1,
            '조언': 1.05,
            '경제': 1.2,
            '노후': 1.1,
            '결혼': 1.1
        }

        # 감지된 상황들의 평균 가중치
        weights = [situation_weights.get(situation, 1.0) for situation in situations]
        return np.mean(weights) if weights else 1.0

    def get_enhanced_text_embedding(self, text, emotions=None, situations=None):
        """
        강화된 텍스트 임베딩 - 감정/상황 정보 포함

        Args:
            text: 입력 텍스트
            emotions: 감지된 감정 리스트
            situations: 감지된 상황 리스트

        Returns:
            enhanced_embedding: 강화된 임베딩 벡터
        """
        try:
            # 기본 텍스트 임베딩
            base_embedding = self.get_text_embedding(text)

            if not emotions and not situations:
                return base_embedding

            # 감정/상황 키워드를 텍스트에 추가하여 재임베딩
            enhanced_text = text

            if emotions:
                emotion_keywords = ' '.join(emotions)
                enhanced_text += f" {emotion_keywords}"

            if situations:
                situation_keywords = ' '.join(situations)
                enhanced_text += f" {situation_keywords}"

            # 강화된 텍스트로 임베딩 생성
            enhanced_embedding = self.get_text_embedding(enhanced_text)

            # 기본 임베딩과 강화된 임베딩을 결합 (7:3 비율)
            combined_embedding = 0.7 * base_embedding + 0.3 * enhanced_embedding

            return combined_embedding

        except Exception as e:
            print(f"[!] 강화된 임베딩 생성 실패: {e}")
            return self.get_text_embedding(text)

    def batch_text_similarity(self, text_emb, image_embeddings, batch_size=100):
        """
        배치 단위로 유사도 계산 (메모리 효율성)

        Args:
            text_emb: 텍스트 임베딩
            image_embeddings: 이미지 임베딩 배열
            batch_size: 배치 크기

        Returns:
            similarities: 유사도 점수 배열
        """
        try:
            n_images = len(image_embeddings)
            similarities = np.zeros(n_images)

            if text_emb.ndim == 1:
                text_emb = text_emb.reshape(1, -1)

            for i in range(0, n_images, batch_size):
                end_idx = min(i + batch_size, n_images)
                batch_embeddings = image_embeddings[i:end_idx]

                batch_similarities = cosine_similarity(text_emb, batch_embeddings)[0]
                similarities[i:end_idx] = batch_similarities

            return np.nan_to_num(similarities, nan=0.0)

        except Exception as e:
            print(f"[!] 배치 유사도 계산 실패: {e}")
            return np.zeros(len(image_embeddings))

    def debug_similarity_calculation(self, text, text_emb, image_embeddings, top_k=5):
        """
        유사도 계산 디버깅 정보 출력

        Args:
            text: 입력 텍스트
            text_emb: 텍스트 임베딩
            image_embeddings: 이미지 임베딩 배열
            top_k: 상위 K개 결과 출력
        """
        try:
            similarities = self._calculate_text_similarity(text_emb, image_embeddings)

            print(f"[DEBUG] 텍스트: {text[:50]}...")
            print(f"[DEBUG] 텍스트 임베딩 차원: {text_emb.shape}")
            print(f"[DEBUG] 이미지 임베딩 차원: {image_embeddings.shape}")
            print(f"[DEBUG] 유사도 범위: {similarities.min():.4f} ~ {similarities.max():.4f}")
            print(f"[DEBUG] 평균 유사도: {similarities.mean():.4f}")

            # 상위 K개 결과
            if len(similarities) > 0:
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                print(f"[DEBUG] 상위 {top_k}개 유사도:")
                for i, idx in enumerate(top_indices, 1):
                    print(f"  {i}. 인덱스 {idx}: {similarities[idx]:.4f}")

        except Exception as e:
            print(f"[!] 디버깅 정보 출력 실패: {e}")

    def encode_text(self, text):
        """get_text_embedding의 별칭"""
        return self.get_text_embedding(text)

    def get_image_embedding(self, image_path):
        """encode_image의 별칭"""
        return self.encode_image(image_path)

    def encode_all_images(self, metadata, image_root):
        embeddings = []
        filenames = []
        for item in metadata:
            image_path = os.path.join(image_root, item['filepath'])
            try:
                emb = self.encode_image(image_path)
                embeddings.append(emb)
                filenames.append(item['filename'])
            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")
        return embeddings, filenames

    def preprocess_korean_dialogue(self, text):
        """한국어 대화 전처리 유틸리티 메서드"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)

        # 특수문자 정리 (한국어, 영어, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?:-]', '', text)

        # 길이 제한 (더 보수적으로)
        if len(text) > 150:
            sentences = text.split('.')
            if len(sentences) > 2:
                text = '.'.join(sentences[:2]) + '.'
            else:
                text = text[:150] + '...'

        return text.strip()

    def test_tokenization(self, text):
        """토큰화 테스트 메서드 (디버깅용)"""
        try:
            processed_text = self.truncate_korean_text(text)
            tokens = clip.tokenize([processed_text], truncate=True)
            print(f"원본 텍스트: {text}")
            print(f"처리된 텍스트: {processed_text}")
            print(f"토큰 수: {tokens.shape[1]}")
            print(f"토큰화 성공: True")
            return True
        except Exception as e:
            print(f"토큰화 실패: {e}")
            return False

    def validate_embeddings(self, text_emb, image_embeddings):
        """임베딩 유효성 검증"""
        issues = []

        # 텍스트 임베딩 검증
        if text_emb is None:
            issues.append("텍스트 임베딩이 None입니다")
        elif len(text_emb) == 0:
            issues.append("텍스트 임베딩이 비어있습니다")
        elif np.any(np.isnan(text_emb)):
            issues.append("텍스트 임베딩에 NaN 값이 있습니다")
        elif np.any(np.isinf(text_emb)):
            issues.append("텍스트 임베딩에 무한대 값이 있습니다")

        # 이미지 임베딩 검증
        if len(image_embeddings) == 0:
            issues.append("이미지 임베딩이 비어있습니다")
        elif np.any(np.isnan(image_embeddings)):
            issues.append("이미지 임베딩에 NaN 값이 있습니다")
        elif np.any(np.isinf(image_embeddings)):
            issues.append("이미지 임베딩에 무한대 값이 있습니다")

        if issues:
            print(f"[!] 임베딩 검증 실패:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        return True

    def get_embedding_stats(self, text_emb, image_embeddings):
        """임베딩 통계 정보 출력"""
        try:
            print(f"[STATS] 텍스트 임베딩:")
            print(f"  - 차원: {text_emb.shape}")
            print(f"  - 평균: {np.mean(text_emb):.4f}")
            print(f"  - 표준편차: {np.std(text_emb):.4f}")
            print(f"  - 범위: {np.min(text_emb):.4f} ~ {np.max(text_emb):.4f}")

            print(f"[STATS] 이미지 임베딩:")
            print(f"  - 차원: {image_embeddings.shape}")
            print(f"  - 평균: {np.mean(image_embeddings):.4f}")
            print(f"  - 표준편차: {np.std(image_embeddings):.4f}")
            print(f"  - 범위: {np.min(image_embeddings):.4f} ~ {np.max(image_embeddings):.4f}")

        except Exception as e:
            print(f"[!] 통계 정보 출력 실패: {e}")

    def similarity_analysis(self, text, emotions, situations, image_embeddings, image_files, top_k=10):
        """
        종합적인 유사도 분석 및 결과 출력

        Args:
            text: 입력 텍스트
            emotions: 감지된 감정 리스트
            situations: 감지된 상황 리스트
            image_embeddings: 이미지 임베딩 배열
            image_files: 이미지 파일 경로 리스트
            top_k: 상위 K개 결과 분석

        Returns:
            analysis_results: 분석 결과 딕셔너리
        """
        try:
            # 텍스트 임베딩 생성
            text_emb = self.get_text_embedding(text)

            # 듀얼 유사도 계산
            combined_similarities, text_similarities, weighted_similarities = self.get_dual_similarities(
                text_emb, image_embeddings, emotions, situations
            )

            # Top-K 분석
            top_indices = np.argsort(combined_similarities)[-top_k:][::-1]

            analysis_results = {
                'input_text': text,
                'emotions': emotions,
                'situations': situations,
                'top_matches': [],
                'similarity_stats': {
                    'combined_avg': float(np.mean(combined_similarities)),
                    'combined_max': float(np.max(combined_similarities)),
                    'combined_min': float(np.min(combined_similarities)),
                    'text_avg': float(np.mean(text_similarities)),
                    'weighted_avg': float(np.mean(weighted_similarities))
                }
            }

            # Top-K 결과 저장
            for i, idx in enumerate(top_indices, 1):
                if idx < len(image_files):
                    analysis_results['top_matches'].append({
                        'rank': i,
                        'image_file': os.path.basename(image_files[idx]),
                        'combined_score': float(combined_similarities[idx]),
                        'text_score': float(text_similarities[idx]),
                        'weighted_score': float(weighted_similarities[idx])
                    })

            return analysis_results

        except Exception as e:
            print(f"[!] 유사도 분석 실패: {e}")
            return {
                'input_text': text,
                'emotions': emotions or [],
                'situations': situations or [],
                'top_matches': [],
                'similarity_stats': {},
                'error': str(e)
            }

    def optimize_similarity_computation(self, text_emb, image_embeddings):
        """
        유사도 계산 최적화 (대용량 데이터용)

        Args:
            text_emb: 텍스트 임베딩
            image_embeddings: 이미지 임베딩 배열

        Returns:
            optimized_similarities: 최적화된 유사도 점수
        """
        try:
            # 메모리 사용량에 따라 배치 크기 조정
            n_images = len(image_embeddings)

            if n_images < 1000:
                # 작은 데이터셋: 한 번에 계산
                return self._calculate_text_similarity(text_emb, image_embeddings)
            elif n_images < 10000:
                # 중간 데이터셋: 배치 처리
                return self.batch_text_similarity(text_emb, image_embeddings, batch_size=500)
            else:
                # 대용량 데이터셋: 작은 배치로 처리
                return self.batch_text_similarity(text_emb, image_embeddings, batch_size=100)

        except Exception as e:
            print(f"[!] 최적화된 유사도 계산 실패: {e}")
            return np.zeros(len(image_embeddings))

    def cache_embeddings(self, texts, cache_size=1000):
        """
        텍스트 임베딩 캐싱 시스템

        Args:
            texts: 텍스트 리스트
            cache_size: 캐시 최대 크기

        Returns:
            embedding_cache: 캐시된 임베딩 딕셔너리
        """
        try:
            import hashlib
            from collections import OrderedDict

            embedding_cache = OrderedDict()

            for text in texts:
                # 텍스트 해시 생성
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

                if text_hash not in embedding_cache:
                    # 캐시 크기 제한
                    if len(embedding_cache) >= cache_size:
                        embedding_cache.popitem(last=False)  # 가장 오래된 항목 제거

                    # 임베딩 생성 및 캐시 저장
                    embedding = self.get_text_embedding(text)
                    embedding_cache[text_hash] = {
                        'text': text,
                        'embedding': embedding
                    }

            return embedding_cache

        except Exception as e:
            print(f"[!] 임베딩 캐싱 실패: {e}")
            return {}

    def compare_similarity_methods(self, text, image_embeddings, emotions=None, situations=None):
        """
        다양한 유사도 계산 방법 비교

        Args:
            text: 입력 텍스트
            image_embeddings: 이미지 임베딩 배열
            emotions: 감정 리스트
            situations: 상황 리스트

        Returns:
            comparison_results: 비교 결과
        """
        try:
            text_emb = self.get_text_embedding(text)

            # 1. 기본 텍스트 유사도
            text_similarities = self._calculate_text_similarity(text_emb, image_embeddings)

            # 2. 강화된 임베딩 유사도
            enhanced_emb = self.get_enhanced_text_embedding(text, emotions, situations)
            enhanced_similarities = self._calculate_text_similarity(enhanced_emb, image_embeddings)

            # 3. 듀얼 유사도
            dual_similarities, _, _ = self.get_dual_similarities(text_emb, image_embeddings, emotions, situations)

            # 4. 배치 유사도 (성능 비교용)
            batch_similarities = self.batch_text_similarity(text_emb, image_embeddings)

            # 결과 비교
            comparison_results = {
                'text_method': {
                    'max_score': float(np.max(text_similarities)),
                    'avg_score': float(np.mean(text_similarities)),
                    'top_index': int(np.argmax(text_similarities))
                },
                'enhanced_method': {
                    'max_score': float(np.max(enhanced_similarities)),
                    'avg_score': float(np.mean(enhanced_similarities)),
                    'top_index': int(np.argmax(enhanced_similarities))
                },
                'dual_method': {
                    'max_score': float(np.max(dual_similarities)),
                    'avg_score': float(np.mean(dual_similarities)),
                    'top_index': int(np.argmax(dual_similarities))
                },
                'batch_method': {
                    'max_score': float(np.max(batch_similarities)),
                    'avg_score': float(np.mean(batch_similarities)),
                    'top_index': int(np.argmax(batch_similarities))
                }
            }

            return comparison_results

        except Exception as e:
            print(f"[!] 유사도 방법 비교 실패: {e}")
            return {}

    def health_check(self):
        """
        CLIPEncoder 상태 점검

        Returns:
            health_status: 상태 점검 결과
        """
        try:
            health_status = {
                'model_loaded': False,
                'device_available': False,
                'tokenization_working': False,
                'embedding_working': False,
                'similarity_working': False,
                'issues': []
            }

            # 1. 모델 로드 확인
            if hasattr(self, 'model') and self.model is not None:
                health_status['model_loaded'] = True
            else:
                health_status['issues'].append("CLIP 모델이 로드되지 않음")

            # 2. 디바이스 확인
            if hasattr(self, 'device'):
                health_status['device_available'] = True
                health_status['device'] = str(self.device)
            else:
                health_status['issues'].append("디바이스가 설정되지 않음")

            # 3. 토큰화 테스트
            try:
                test_result = self.test_tokenization("테스트 텍스트")
                health_status['tokenization_working'] = test_result
                if not test_result:
                    health_status['issues'].append("토큰화 테스트 실패")
            except:
                health_status['issues'].append("토큰화 테스트 중 예외 발생")

            # 4. 임베딩 생성 테스트
            try:
                test_embedding = self.get_text_embedding("테스트 텍스트")
                if test_embedding is not None and len(test_embedding) > 0:
                    health_status['embedding_working'] = True
                else:
                    health_status['issues'].append("임베딩 생성 실패")
            except:
                health_status['issues'].append("임베딩 생성 중 예외 발생")

            # 5. 유사도 계산 테스트
            try:
                test_emb = np.random.rand(512)
                test_img_embs = np.random.rand(5, 512)
                similarities = self._calculate_text_similarity(test_emb, test_img_embs)
                if len(similarities) == 5:
                    health_status['similarity_working'] = True
                else:
                    health_status['issues'].append("유사도 계산 결과 차원 오류")
            except:
                health_status['issues'].append("유사도 계산 중 예외 발생")

            # 전체 상태 평가
            working_components = sum([
                health_status['model_loaded'],
                health_status['device_available'],
                health_status['tokenization_working'],
                health_status['embedding_working'],
                health_status['similarity_working']
            ])

            health_status['overall_health'] = working_components / 5.0
            health_status['status'] = 'HEALTHY' if working_components >= 4 else 'ISSUES_DETECTED'

            return health_status

        except Exception as e:
            return {
                'status': 'CRITICAL_ERROR',
                'error': str(e),
                'issues': [f"상태 점검 중 치명적 오류: {e}"]
            }

    def print_health_report(self):
        """상태 점검 결과 출력"""
        health = self.health_check()

        print("\n" + "=" * 50)
        print("🔍 CLIPEncoder 상태 점검 결과")
        print("=" * 50)

        print(f"전체 상태: {health['status']}")
        print(f"건강도: {health.get('overall_health', 0):.1%}")

        print(f"\n구성 요소 상태:")
        print(f"  ✅ 모델 로드: {'OK' if health['model_loaded'] else '❌ FAIL'}")
        print(f"  ✅ 디바이스: {'OK' if health['device_available'] else '❌ FAIL'}")
        if 'device' in health:
            print(f"     디바이스: {health['device']}")
        print(f"  ✅ 토큰화: {'OK' if health['tokenization_working'] else '❌ FAIL'}")
        print(f"  ✅ 임베딩: {'OK' if health['embedding_working'] else '❌ FAIL'}")
        print(f"  ✅ 유사도: {'OK' if health['similarity_working'] else '❌ FAIL'}")

        if health['issues']:
            print(f"\n⚠️ 발견된 문제:")
            for issue in health['issues']:
                print(f"  - {issue}")
        else:
            print(f"\n🎉 모든 구성 요소가 정상 작동 중!")

        print("=" * 50)