from PIL import Image
import torch
import clip
import os
import re
import numpy as np


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

            return text_features.cpu().numpy()[0]

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