from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os


class EnhancedMatcher:
    def __init__(self, metadata_file="data/enhanced_image_metadata.json"):
        # 이미지 메타데이터 로드
        self.metadata = []
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"[*] 메타데이터 로드 완료: {len(self.metadata)}개 이미지")
            except Exception as e:
                print(f"[!] 메타데이터 로드 실패: {e}")

        # 감정/상황별 키워드 매핑
        self.emotion_keywords = {
            '화나다': ['화', '짜증', '빡', '분노', '열받', '기막', '분하다', '성나다'],
            '슬프다': ['슬픈', '우울', '속상', '서럽', '눈물', '울고', '서글'],
            '기쁘다': ['기쁜', '행복', '좋다', '신나', '즐거', '만족', '기뻐'],
            '무섭다': ['무서', '걱정', '불안', '두려', '무서워', '걱정되'],
            '놀랍다': ['놀라', '신기', '대박', '헐', '와', '당황'],
            '싫다': ['싫어', '역겨', '혐오', '별로', '안좋'],
            '감사하다': ['감사', '고마워', '다행', '고맙다', '감사해'],
            '편안하다': ['편안', '안정', '평온', '여유', '릴렉스'],
            '자신하다': ['자신', '확신', '자부심', '자랑', '당당']
        }

        self.situation_keywords = {
            '직장': ['회사', '직장', '상사', '동료', '업무', '일', '근무', '직장인', '회사원'],
            '연애': ['연인', '남친', '여친', '애인', '데이트', '사랑'],
            '돈': ['돈', '월급', '급여', '지출', '소비', '비용', '경제'],
            '인간관계': ['친구', '사람', '관계', '만남', '소통'],
            '스트레스': ['스트레스', '피곤', '힘들', '지쳐', '부담'],
            '축하': ['축하', '생일', '승진', '성공', '합격'],
            '조언': ['조언', '상담', '도움', '충고'],
            '업무': ['업무', '일', '프로젝트', '회의'],
            '개인적': ['개인', '혼자', '나만', '내가']
        }

    def analyze_text_features(self, text):
        """텍스트에서 감정과 상황 키워드 추출"""
        detected_emotions = []
        detected_situations = []

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

        return detected_emotions, detected_situations

    def find_metadata_by_filename(self, filename):
        """파일명으로 메타데이터 찾기"""
        for item in self.metadata:
            if item['filename'] == filename:
                return item
        return None

    def calculate_metadata_bonus(self, text_emotions, text_situations, image_filename, text):
        """메타데이터 기반 보너스 점수 계산"""
        metadata = self.find_metadata_by_filename(image_filename)
        if not metadata:
            return 1.0

        bonus = 1.0

        # 1. 카테고리 매칭 (감정 카테고리)
        category = metadata.get('category', '')
        subcategory = metadata.get('subcategory', '')

        for emotion in text_emotions:
            if emotion == '기쁘다' and category == '기쁨':
                bonus += 0.4
            elif emotion == '슬프다' and category == '슬픔':
                bonus += 0.4
            elif emotion == '화나다' and category == '화남':
                bonus += 0.4
            elif emotion == '불안하다' and category == '불안':
                bonus += 0.4
            elif emotion == '감사하다' and subcategory == '감사하는':
                bonus += 0.3
            elif emotion == '자신하다' and subcategory == '자신하는':
                bonus += 0.3

        # 2. 태그 매칭
        tags = metadata.get('tags', [])
        for emotion in text_emotions:
            emotion_base = emotion.replace('다', '')  # '기쁘다' -> '기쁘'
            for tag in tags:
                if emotion_base in tag or tag in emotion:
                    bonus += 0.2
                    break

        # 3. 사용 맥락 매칭
        usage_context = metadata.get('usage_context', [])
        for situation in text_situations:
            if situation in usage_context:
                bonus += 0.25

        # 4. 텍스트 설명에서 키워드 매칭
        description = metadata.get('text_description', '').lower()
        for emotion in text_emotions:
            if emotion.replace('다', '') in description:
                bonus += 0.15

        # 5. 직접적인 키워드 매칭 (텍스트 내용과 설명)
        text_words = text.replace('.', ' ').replace(',', ' ').split()
        for word in text_words:
            if len(word) > 1 and word in description:
                bonus += 0.1
                break

        return min(bonus, 2.5)  # 최대 2.5배까지

    def find_best_match(self, text_emb, image_embs, image_files, text):
        """향상된 매칭 알고리즘"""
        # 기본 코사인 유사도 계산
        similarities = cosine_similarity([text_emb], image_embs)[0]

        # 텍스트 특성 분석
        text_emotions, text_situations = self.analyze_text_features(text)

        # 메타데이터 기반 보너스 적용
        enhanced_similarities = []
        for i, sim in enumerate(similarities):
            image_filename = os.path.basename(image_files[i])
            bonus = self.calculate_metadata_bonus(text_emotions, text_situations, image_filename, text)
            enhanced_score = sim * bonus
            enhanced_similarities.append(enhanced_score)

        enhanced_similarities = np.array(enhanced_similarities)

        # 최고 매치 찾기
        best_idx = np.argmax(enhanced_similarities)
        best_score = enhanced_similarities[best_idx]

        # 상위 5개 결과
        top_5_indices = np.argsort(enhanced_similarities)[-5:][::-1]
        top_5_results = []
        for i in top_5_indices:
            image_filename = os.path.basename(image_files[i])
            metadata = self.find_metadata_by_filename(image_filename)

            result_info = {
                'filename': image_filename,
                'score': enhanced_similarities[i],
                'original_similarity': similarities[i],
                'category': metadata.get('category', '알 수 없음') if metadata else '알 수 없음',
                'subcategory': metadata.get('subcategory', '알 수 없음') if metadata else '알 수 없음'
            }
            top_5_results.append(result_info)

        return best_idx, best_score, top_5_results, text_emotions, text_situations