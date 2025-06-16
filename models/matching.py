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
                    '어이없', '답답', '스트레스', '빡쳐', '열나', '빈정', '약올라'],

            '슬프다': ['슬픈', '우울', '속상', '서럽', '눈물', '울고', '서글',
                    '슬퍼', '우울해', '서러워', '눈물나', '마음아픈', '가슴아픈',
                    '쓸쓸', '외로워', '허전', '헤어져', '이별', '그리워',
                    '실망', '실망스러', '서운', '안타까', '가슴', '마음', '상처'],

            '기쁘다': ['기쁜', '행복', '좋다', '신나', '즐거', '만족', '기뻐',
                    '좋아', '행복해', '기분좋', '신이나', '즐거워', '만족해',
                    '성공', '승진', '합격', '축하', '기뻐요', '좋네', '최고',
                    '설레', '설렘', '들뜨', '기대', '환상', '완벽', '멋져'],

            '무섭다': ['무서', '걱정', '불안', '두려', '무서워', '걱정되',
                    '불안해', '두려워', '무시무시', '심란', '조마조마',
                    '걱정스러', '염려', '근심', '우려', '무서운'],

            '놀랍다': ['놀라', '신기', '대박', '헐', '와', '당황', '어쩌지',
                    '놀라워', '신기해', '당황스러', '어리둥절', '깜짝',
                    '놀란', '충격', '경악'],

            '싫다': ['싫어', '역겨', '혐오', '별로', '안좋', '짜증',
                   '싫다', '역겹', '혐오스러', '꼴보기싫', '지겨워',
                   '질린', '지친', '피곤'],

            '감사하다': ['감사', '고마워', '다행', '고맙다', '감사해', '고마운',
                     '감사드려', '도움', '고마웠', '감사합니다', '고마워요',
                     '든든', '응원', '믿어'],

            '편안하다': ['편안', '안정', '평온', '여유', '릴렉스', '차분',
                     '안락', '평화', '느긋', '안심', '편해', '가벼운'],

            '자신하다': ['자신', '확신', '자부심', '자랑', '당당', '뿌듯',
                     '자신있', '확신해', '자부', '당당해', '늘고', '늘어',
                     '발전', '성장', '진전', '향상', '나아', '좋아']
        }

        # 상황 키워드 확장
        self.situation_keywords = {
            '직장': ['회사', '직장', '상사', '동료', '업무', '일', '근무', '직장인', '회사원',
                   '막내', '팀', '프로젝트', '회의', '야근', '퇴사', '이직'],
            '연애': ['연인', '남친', '여친', '애인', '데이트', '사랑', '헤어져', '이별',
                   '커플', '만남', '결혼', '연애'],
            '돈': ['돈', '월급', '급여', '지출', '소비', '비용', '경제', '깎였', '줄어',
                  '보너스', '생활비', '물가', '절약', '금전'],
            '인간관계': ['친구', '사람', '관계', '만남', '소통', '가족', '부모', '형제',
                     '동생', '언니', '오빠', '누나', '협업', '팀원'],
            '스트레스': ['스트레스', '피곤', '힘들', '지쳐', '부담', '압박', '고민'],
            '축하': ['축하', '생일', '승진', '성공', '합격', '결혼', '기념일'],
            '조언': ['조언', '상담', '도움', '충고', '응원', '지지'],
            '업무': ['업무', '일', '프로젝트', '회의', '과제', '작업', '개발', '계획'],
            '개인적': ['개인', '혼자', '나만', '내가', '스스로', '자신', '본인',
                    '배우', '학습', '공부', '시험', '이사', '건강', '운동', '취미',
                    '성장', '발전', '기술', '능력', '실력', '새로운', '도시', '변화']
        }

    def analyze_emotions(self, text):
        """감정 분석 - 개선된 버전"""
        if not text:
            return []

        detected_emotions = []
        text_lower = text.lower()

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_emotions.append(emotion)
                    break  # 감정당 한 번만 추가

        return detected_emotions

    def analyze_situations(self, text):
        """상황 분석 - 개선된 버전"""
        if not text:
            return []

        detected_situations = []
        text_lower = text.lower()

        for situation, keywords in self.situation_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_situations.append(situation)
                    break  # 상황당 한 번만 추가

        return detected_situations

    def analyze_text_features(self, text):
        """향상된 텍스트 분석"""
        try:
            detected_emotions = self.analyze_emotions(text)
            detected_situations = self.analyze_situations(text)

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
        """메타데이터 기반 보너스 점수 계산"""
        try:
            metadata = self.find_metadata_by_filename(image_filename)
            if not metadata:
                return 1.0

            bonus = 1.0

            # 부정적 맥락에서 기쁨 이미지 패널티
            if self.detect_negative_context(text) and metadata.get('category') == '기쁨':
                bonus *= 0.2

            # 카테고리 매칭
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

            return min(bonus, 3.0)

        except Exception as e:
            print(f"[!] 보너스 계산 예외 ({image_filename}): {e}")
            return 1.0

    def find_best_match_hybrid(self, encoder, text, image_embs, image_files):
        """
        ✅ 수정된 하이브리드 매칭 - CLIPEncoder와 호환

        Args:
            encoder: CLIPEncoder 또는 HybridEncoder 인스턴스
            text: 입력 텍스트 (문자열)
            image_embs: 이미지 임베딩 배열
            image_files: 이미지 파일 경로 리스트
        """
        try:
            print(f"[DEBUG] 이미지 임베딩 차원: {image_embs.shape}")

            # 🔧 텍스트 임베딩 생성 (문자열을 임베딩으로 변환)
            if hasattr(encoder, 'get_text_embedding'):
                # CLIPEncoder 사용
                text_emb = encoder.get_text_embedding(text)
                print(f"[DEBUG] CLIP 텍스트 임베딩 차원: {len(text_emb)}")

                # 듀얼 유사도 시도 (있으면)
                if hasattr(encoder, 'get_dual_similarities'):
                    try:
                        # 감정/상황 분석
                        emotions, situations = self.analyze_text_features(text)

                        # ✅ 임베딩을 전달 (문자열이 아님!)
                        combined_similarities, text_similarities, weighted_similarities = encoder.get_dual_similarities(
                            text_emb, image_embs, emotions, situations
                        )
                        print("[DEBUG] 듀얼 유사도 계산 성공")
                        final_similarities = combined_similarities

                    except Exception as e:
                        print(f"[!] 듀얼 유사도 실패: {e}")
                        # 기본 유사도 계산
                        final_similarities = cosine_similarity([text_emb], image_embs)[0]

                else:
                    # 기본 유사도 계산
                    final_similarities = cosine_similarity([text_emb], image_embs)[0]

            elif hasattr(encoder, 'get_dual_similarities'):
                # HybridEncoder 사용
                emotions, situations = self.analyze_text_features(text)
                final_similarities, _, _ = encoder.get_dual_similarities(text, image_embs)
                print("[DEBUG] HybridEncoder 듀얼 유사도 사용")

            else:
                # 폴백: 기본 CLIP 방식
                if hasattr(encoder, 'clip_encoder'):
                    text_emb = encoder.clip_encoder.get_text_embedding(text)
                else:
                    text_emb = encoder.encode_text(text)  # 다른 인터페이스 시도

                final_similarities = cosine_similarity([text_emb], image_embs)[0]
                print("[DEBUG] 폴백 유사도 계산 사용")

            # 텍스트 분석 (한 번만 수행)
            emotions, situations = self.analyze_text_features(text)

            # 부정적 맥락 보정
            if self.detect_negative_context(text):
                for i, img_file in enumerate(image_files):
                    metadata = self.find_metadata_by_filename(os.path.basename(img_file))
                    if metadata and metadata.get('category', '') == '기쁨':
                        final_similarities[i] *= 0.2

            # 메타데이터 보너스 적용
            enhanced_similarities = []
            for i, sim in enumerate(final_similarities):
                try:
                    image_filename = os.path.basename(image_files[i])
                    bonus = self.calculate_metadata_bonus(emotions, situations, image_filename, text)
                    enhanced_score = sim * bonus
                    enhanced_similarities.append(enhanced_score)
                except Exception as e:
                    print(f"[!] 보너스 계산 실패 (이미지 {i}): {e}")
                    enhanced_similarities.append(sim)

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
                        'score': float(enhanced_similarities[i]),
                        'combined_sim': float(final_similarities[i]),
                        'category': metadata.get('category', '알 수 없음') if metadata else '알 수 없음',
                        'subcategory': metadata.get('subcategory', '알 수 없음') if metadata else '알 수 없음'
                    }
                    top_5_results.append(result_info)
                except Exception as e:
                    print(f"[!] Top 5 결과 생성 실패 (인덱스 {i}): {e}")

            return best_idx, float(best_score), top_5_results, emotions, situations

        except Exception as e:
            print(f"[!] 하이브리드 매칭 전체 실패: {e}")
            # 최종 폴백
            return 0, 0.0, [], ["오류"], ["오류상황"]


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