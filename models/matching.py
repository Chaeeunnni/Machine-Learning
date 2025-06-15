from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os


class HybridMatcher:
    def __init__(self, metadata_file="data/enhanced_image_metadata.json"):
        # 메타데이터 로드
        self.metadata = []
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"[*] 메타데이터 로드 완료: {len(self.metadata)}개 이미지")
            except Exception as e:
                print(f"[!] 메타데이터 로드 실패: {e}")

        # 확장된 감정 키워드
        self.emotion_keywords = {
            '화나다': ['화', '짜증', '빡', '분노', '열받', '기막', '분하다', '성나다',
                    '분해', '약오르', '빡치', '열불나', '속터져', '골치',
                    '거슬려', '짜증나', '화나', '열받아', '분하고', '섭섭해', '억울',
                    '어이없', '답답', '스트레스'],
            '슬프다': ['슬픈', '우울', '속상', '서럽', '눈물', '울고', '서글',
                    '슬퍼', '우울해', '서러워', '눈물나', '마음아픈', '가슴아픈',
                    '쓸쓸', '외로워', '허전', '헤어져', '이별', '그리워'],
            '기쁘다': ['기쁜', '행복', '좋다', '신나', '즐거', '만족', '기뻐',
                    '좋아', '행복해', '기분좋', '신이나', '즐거워', '만족해',
                    '성공', '승진', '합격', '축하', '기뻐요', '좋네', '최고'],
            '무섭다': ['무서', '걱정', '불안', '두려', '무서워', '걱정되',
                    '불안해', '두려워', '무시무시', '심란', '조마조마'],
            '놀랍다': ['놀라', '신기', '대박', '헐', '와', '당황', '어쩌지',
                    '놀라워', '신기해', '당황스러', '어리둥절', '깜짝'],
            '싫다': ['싫어', '역겨', '혐오', '별로', '안좋', '짜증',
                   '싫다', '역겹', '혐오스러', '꼴보기싫', '지겨워'],
            '감사하다': ['감사', '고마워', '다행', '고맙다', '감사해', '고마운',
                     '감사드려', '도움', '고마웠', '감사합니다'],
            '편안하다': ['편안', '안정', '평온', '여유', '릴렉스', '차분',
                     '안락', '평화', '느긋'],
            '자신하다': ['자신', '확신', '자부심', '자랑', '당당', '뿌듯',
                     '자신있', '확신해', '자부', '당당해']
        }

        self.situation_keywords = {
            '직장': ['회사', '직장', '상사', '동료', '업무', '일', '근무', '직장인', '회사원', '막내'],
            '연애': ['연인', '남친', '여친', '애인', '데이트', '사랑', '헤어져', '이별'],
            '돈': ['돈', '월급', '급여', '지출', '소비', '비용', '경제', '깎였', '줄어'],
            '인간관계': ['친구', '사람', '관계', '만남', '소통'],
            '스트레스': ['스트레스', '피곤', '힘들', '지쳐', '부담'],
            '축하': ['축하', '생일', '승진', '성공', '합격'],
            '조언': ['조언', '상담', '도움', '충고'],
            '업무': ['업무', '일', '프로젝트', '회의'],
            '개인적': ['개인', '혼자', '나만', '내가']
        }

    def analyze_text_features(self, text):
        """텍스트에서 감정과 상황 키워드 추출 - 완전히 수정"""
        detected_emotions = []
        detected_situations = []

        try:
            # 감정 키워드 검출
            for emotion, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        detected_emotions.append(emotion)
                        break

            # 상황 키워드 검출
            for situation, keywords in self.situation_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        detected_situations.append(situation)
                        break

            print(f"[DEBUG] 텍스트 분석 결과 - 감정: {detected_emotions}, 상황: {detected_situations}")
            return detected_emotions, detected_situations

        except Exception as e:
            print(f"[!] 텍스트 분석 예외: {e}")
            return [], []

    def detect_negative_context(self, text):
        """부정적 맥락 감지"""
        negative_indicators = [
            '깎였', '줄어', '실패', '망했', '안돼', '문제', '힘들',
            '어려워', '못하겠', '싫어', '나빠', '최악', '별로',
            '거슬려', '스트레스', '피곤', '지쳐', '힘들어', '분하고', '섭섭'
        ]
        return any(indicator in text for indicator in negative_indicators)

    def find_metadata_by_filename(self, filename):
        """파일명으로 메타데이터 찾기"""
        for item in self.metadata:
            if item['filename'] == filename:
                return item
        return None

    def calculate_metadata_bonus(self, text_emotions, text_situations, image_filename, text):
        """메타데이터 기반 보너스 점수 계산 - 완전히 수정"""
        try:
            metadata = self.find_metadata_by_filename(image_filename)
            if not metadata:
                print(f"[DEBUG] 메타데이터 없음: {image_filename}")
                return 1.0

            bonus = 1.0

            # 부정적 맥락에서 기쁨 이미지 패널티
            if self.detect_negative_context(text) and metadata.get('category') == '기쁨':
                bonus *= 0.2

            # 1. 카테고리 매칭
            category = metadata.get('category', '')
            subcategory = metadata.get('subcategory', '')

            for emotion in text_emotions:
                if emotion == '기쁘다' and category == '기쁨':
                    bonus += 0.8
                elif emotion == '슬프다' and category == '슬픔':
                    bonus += 0.8
                elif emotion == '화나다' and category in ['분노', '당황']:
                    bonus += 0.8
                elif emotion == '불안하다' and category == '불안':
                    bonus += 0.8
                elif emotion == '감사하다' and subcategory == '감사하는':
                    bonus += 0.6
                elif emotion == '자신하다' and subcategory == '자신하는':
                    bonus += 0.6

            # 2. 태그 매칭
            tags = metadata.get('tags', [])
            for emotion in text_emotions:
                emotion_base = emotion.replace('다', '')
                for tag in tags:
                    if emotion_base in tag or tag in emotion:
                        bonus += 0.3
                        break

            # 3. 사용 맥락 매칭
            usage_context = metadata.get('usage_context', [])
            for situation in text_situations:
                if situation in usage_context:
                    bonus += 0.4

            # 4. 텍스트 설명에서 키워드 매칭
            description = metadata.get('text_description', '').lower()
            for emotion in text_emotions:
                if emotion.replace('다', '') in description:
                    bonus += 0.2

            # 5. 직접적인 키워드 매칭
            text_words = text.replace('.', ' ').replace(',', ' ').split()
            for word in text_words:
                if len(word) > 1 and word in description:
                    bonus += 0.15
                    break

            return min(bonus, 3.0)

        except Exception as e:
            print(f"[!] 보너스 계산 예외 ({image_filename}): {e}")
            return 1.0  # 기본값 반환

    def find_best_match_hybrid(self, hybrid_encoder, text, image_embs, image_files):
        """하이브리드 방식으로 최적 매치 찾기 - 안전화 버전"""

        # 기본값 초기화
        combined_similarities = None
        dual_similarities = None
        kobert_sims = None
        clip_sims = None

        try:
            print(f"[DEBUG] 이미지 임베딩 차원: {image_embs.shape}")

            # 방법 1: 결합된 임베딩 사용
            try:
                combined_text_emb = hybrid_encoder.get_text_embedding(text)
                print(f"[DEBUG] 결합 텍스트 임베딩 차원: {len(combined_text_emb)}")
                combined_similarities = cosine_similarity([combined_text_emb], image_embs)[0]
                print("[DEBUG] 결합 임베딩 유사도 계산 성공")
            except Exception as e:
                print(f"[!] 결합 임베딩 실패: {e}")

            # 방법 2: 각각 계산 후 결합
            try:
                dual_similarities, kobert_sims, clip_sims = hybrid_encoder.get_dual_similarities(text, image_embs)
                print("[DEBUG] 듀얼 유사도 계산 성공")
            except Exception as e:
                print(f"[!] 듀얼 유사도 실패: {e}")

            # 최종 유사도 결정
            if combined_similarities is not None and dual_similarities is not None:
                final_similarities = 0.7 * combined_similarities + 0.3 * dual_similarities
                print("[DEBUG] 하이브리드 방식 사용")
            elif combined_similarities is not None:
                final_similarities = combined_similarities
                print("[DEBUG] 결합 임베딩만 사용")
            elif dual_similarities is not None:
                final_similarities = dual_similarities
                print("[DEBUG] 듀얼 유사도만 사용")
            else:
                print("[!] 모든 방식 실패, CLIP 폴백 사용")
                clip_text_emb = hybrid_encoder.get_clip_text_embedding(text)
                final_similarities = cosine_similarity([clip_text_emb], image_embs)[0]
                kobert_sims = np.zeros_like(final_similarities)
                clip_sims = final_similarities

        except Exception as e:
            print(f"[!] 전체 유사도 계산 실패: {e}")
            try:
                clip_text_emb = hybrid_encoder.clip_encoder.get_text_embedding(text)
                final_similarities = cosine_similarity([clip_text_emb], image_embs)[0]
                kobert_sims = np.zeros_like(final_similarities)
                clip_sims = final_similarities
            except Exception as final_e:
                print(f"[!] 최종 폴백도 실패: {final_e}")
                final_similarities = np.random.random(len(image_files))
                kobert_sims = np.zeros_like(final_similarities)
                clip_sims = np.zeros_like(final_similarities)

        # 텍스트 분석 (반드시 튜플 반환 보장)
        result = self.analyze_text_features(text)
        if result is None or len(result) != 2:
            text_emotions, text_situations = [], []
        else:
            text_emotions, text_situations = result

        # 부정적 맥락 보정
        try:
            if self.detect_negative_context(text):
                for i, img_file in enumerate(image_files):
                    metadata = self.find_metadata_by_filename(os.path.basename(img_file))
                    if metadata and metadata.get('category', '') == '기쁨':
                        final_similarities[i] *= 0.2
        except Exception as e:
            print(f"[!] 부정적 맥락 보정 실패: {e}")

        # 메타데이터 보너스 적용 (안전하게)
        enhanced_similarities = []
        for i, sim in enumerate(final_similarities):
            try:
                image_filename = os.path.basename(image_files[i])
                bonus = self.calculate_metadata_bonus(text_emotions, text_situations, image_filename, text)
                if bonus is None:
                    bonus = 1.0
                enhanced_score = sim * bonus
                enhanced_similarities.append(enhanced_score)
            except Exception as e:
                print(f"[!] 보너스 계산 실패 (이미지 {i}): {e}")
                enhanced_similarities.append(sim)  # 원본 유사도 사용

        enhanced_similarities = np.array(enhanced_similarities)

        # 최고 매치 찾기
        best_idx = np.argmax(enhanced_similarities)
        best_score = enhanced_similarities[best_idx]

        # Top 5 결과
        top_5_indices = np.argsort(enhanced_similarities)[-5:][::-1]
        top_5_results = []
        for i in top_5_indices:
            try:
                image_filename = os.path.basename(image_files[i])
                metadata = self.find_metadata_by_filename(image_filename)

                result_info = {
                    'filename': image_filename,
                    'score': enhanced_similarities[i],
                    'combined_sim': combined_similarities[i] if combined_similarities is not None else 0,
                    'dual_sim': dual_similarities[i] if dual_similarities is not None else 0,
                    'kobert_sim': kobert_sims[i] if kobert_sims is not None else 0,
                    'clip_sim': clip_sims[i] if clip_sims is not None else 0,
                    'category': metadata.get('category', '알 수 없음') if metadata else '알 수 없음',
                    'subcategory': metadata.get('subcategory', '알 수 없음') if metadata else '알 수 없음'
                }
                top_5_results.append(result_info)
            except Exception as e:
                print(f"[!] Top 5 결과 생성 실패 (인덱스 {i}): {e}")

        return best_idx, best_score, top_5_results, text_emotions, text_situations


# 테스트용 메인 함수
if __name__ == "__main__":
    # 매처 테스트
    matcher = HybridMatcher()

    # 텍스트 분석 테스트
    test_texts = [
        "오늘 기분이 좋아!",
        "일이 너무 많아서 스트레스 받아.",
        "친구가 도움을 줘서 감사해."
    ]

    for text in test_texts:
        emotions, situations = matcher.analyze_text_features(text)
        print(f"\n텍스트: {text}")
        print(f"감정: {emotions}, 상황: {situations}")