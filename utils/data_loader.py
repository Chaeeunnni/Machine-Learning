import json
import os
import numpy as np
from sklearn.model_selection import train_test_split


class DialogueDataLoader:
    def __init__(self, dialogue_metadata_file="data/converted_dialogues.json",
                 image_metadata_file="data/image_metadata.json"):
        self.dialogue_data = []
        self.image_data = []

        # 대화 메타데이터 로드
        if os.path.exists(dialogue_metadata_file):
            with open(dialogue_metadata_file, 'r', encoding='utf-8') as f:
                self.dialogue_data = json.load(f)
            print(f"[*] 대화 데이터 로드: {len(self.dialogue_data)}개")
        else:
            print(f"[!] 대화 메타데이터 파일을 찾을 수 없습니다: {dialogue_metadata_file}")

        # 이미지 메타데이터 로드
        if os.path.exists(image_metadata_file):
            with open(image_metadata_file, 'r', encoding='utf-8') as f:
                self.image_data = json.load(f)
            print(f"[*] 이미지 데이터 로드: {len(self.image_data)}개")
        else:
            print(f"[!] 이미지 메타데이터 파일을 찾을 수 없습니다: {image_metadata_file}")

    def get_training_pairs(self):
        """학습용 (텍스트, 이미지) 쌍 생성"""
        training_pairs = []

        for dialogue in self.dialogue_data:
            # 전체 대화 컨텍스트 구성
            full_context = " ".join(dialogue.get('context', []))
            utterance = dialogue.get('utterance', '')
            full_text = f"{full_context} {utterance}".strip()

            # 대화 정보
            dialogue_emotion = dialogue.get('emotion', '')
            dialogue_situation = dialogue.get('situation', [])
            dialogue_tags = dialogue.get('image_tags', [])

            # 매칭되는 이미지들 찾기
            matching_images = self.find_matching_images(
                dialogue_emotion, dialogue_situation, dialogue_tags
            )

            for img_info in matching_images:
                training_pairs.append({
                    'dialogue_id': dialogue.get('dialogue_id'),
                    'text': full_text,
                    'emotion': dialogue_emotion,
                    'situation': dialogue_situation,
                    'tags': dialogue_tags,
                    'image_filename': img_info['filename'],
                    'image_path': img_info['path'],
                    'image_category': img_info['category'],
                    'image_subcategory': img_info['subcategory'],
                    'relevance_score': img_info['relevance_score']
                })

        print(f"[*] 학습 쌍 생성 완료: {len(training_pairs)}개")
        return training_pairs

    def find_matching_images(self, emotion, situations, tags):
        """감정, 상황, 태그 기반으로 매칭되는 이미지 찾기"""
        matching_images = []

        # 감정 매핑 테이블
        emotion_mapping = {
            '분노': ['분노', '화남', '짜증', '노여워'],
            '기쁨': ['기쁨', '행복', '즐거움', '만족', '감사'],
            '슬픔': ['슬픔', '우울', '상처', '서러움'],
            '불안': ['불안', '걱정', '두려움', '당황'],
            '놀람': ['놀람', '당황', '충격'],
            '혐오': ['혐오', '싫음', '역겨움']
        }

        # 상황 매핑 테이블
        situation_mapping = {
            '직장': ['업무', '직장', '회사'],
            '진로': ['업무', '직장', '진로'],
            '취업': ['업무', '직장', '진로'],
            '연애': ['연애', '사랑', '친구'],
            '인간관계': ['친구', '관계', '소통'],
            '가족': ['가족', '친구'],
            '돈': ['경제', '돈', '생활']
        }

        for img in self.image_data:
            if img.get('processing_status') != 'success':
                continue

            relevance_score = 0.0

            # 1. 감정 매칭 (가장 중요한 요소)
            img_category = img.get('category', '').lower()
            img_subcategory = img.get('subcategory', '').lower()
            img_tags = [tag.lower() for tag in img.get('tags', [])]

            # 직접 감정 매칭
            if emotion.lower() in img_category or emotion.lower() in img_subcategory:
                relevance_score += 2.0

            # 감정 매핑 테이블을 통한 매칭
            emotion_keywords = emotion_mapping.get(emotion, [])
            for keyword in emotion_keywords:
                if keyword in img_category or keyword in img_subcategory:
                    relevance_score += 1.5
                    break

            # 2. 태그 매칭
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower in img_tags:
                    relevance_score += 1.0
                elif any(tag_lower in img_tag for img_tag in img_tags):
                    relevance_score += 0.8

            # 3. 서브카테고리와 태그 매칭
            for tag in tags:
                if tag.lower() in img_subcategory:
                    relevance_score += 0.8

            # 4. 상황 매칭
            img_context = [ctx.lower() for ctx in img.get('usage_context', [])]
            for situation in situations:
                situation_keywords = situation_mapping.get(situation, [situation])
                for keyword in situation_keywords:
                    if keyword.lower() in img_context:
                        relevance_score += 0.6
                        break

            # 5. 텍스트 설명에서 키워드 매칭
            description = img.get('text_description', '').lower()

            # 감정 관련 키워드가 설명에 있는지 확인
            for keyword in emotion_keywords:
                if keyword in description:
                    relevance_score += 0.3
                    break

            # 태그가 설명에 있는지 확인
            for tag in tags:
                if tag.lower() in description:
                    relevance_score += 0.2

            # 관련성이 있는 이미지만 포함 (최소 임계값)
            if relevance_score >= 0.5:
                img_path = img.get('processed_path') or img.get('filepath', '')

                # 경로 정규화
                if img_path and not os.path.exists(img_path):
                    # 상대 경로로 시도
                    relative_path = f"data/images/{img.get('filepath', '')}"
                    if os.path.exists(relative_path):
                        img_path = relative_path

                if img_path and os.path.exists(img_path):
                    matching_images.append({
                        'filename': img['filename'],
                        'path': img_path,
                        'category': img.get('category', ''),
                        'subcategory': img.get('subcategory', ''),
                        'tags': img.get('tags', []),
                        'relevance_score': relevance_score
                    })

        # 관련성 점수로 정렬 (내림차순)
        matching_images.sort(key=lambda x: x['relevance_score'], reverse=True)

        # 상위 이미지들만 반환 (너무 많으면 상위 10개)
        max_images = min(10, len(matching_images))
        return matching_images[:max_images]

    def split_data(self, test_size=0.2, random_state=42):
        """데이터를 학습/테스트로 분할"""
        training_pairs = self.get_training_pairs()

        if len(training_pairs) == 0:
            print("[!] 학습 데이터가 없습니다. 메타데이터를 확인해주세요.")
            return [], []

        # 대화 ID별로 그룹화하여 분할 (같은 대화가 train/test에 동시에 들어가지 않도록)
        dialogue_groups = {}
        for pair in training_pairs:
            dialogue_id = pair['dialogue_id']
            if dialogue_id not in dialogue_groups:
                dialogue_groups[dialogue_id] = []
            dialogue_groups[dialogue_id].append(pair)

        # 대화 ID 리스트
        dialogue_ids = list(dialogue_groups.keys())

        # 대화 ID 기준으로 train/test 분할
        train_ids, test_ids = train_test_split(
            dialogue_ids, test_size=test_size, random_state=random_state
        )

        # 분할된 ID에 따라 실제 데이터 분할
        train_data = []
        test_data = []

        for dialogue_id in train_ids:
            train_data.extend(dialogue_groups[dialogue_id])

        for dialogue_id in test_ids:
            test_data.extend(dialogue_groups[dialogue_id])

        print(f"[*] 데이터 분할 완료")
        print(f"    - 학습 대화: {len(train_ids)}개, 학습 쌍: {len(train_data)}개")
        print(f"    - 테스트 대화: {len(test_ids)}개, 테스트 쌍: {len(test_data)}개")

        return train_data, test_data

    def get_statistics(self):
        """데이터 통계 정보 반환"""
        if not self.dialogue_data:
            return {}

        # 감정 분포
        emotion_counts = {}
        situation_counts = {}

        for dialogue in self.dialogue_data:
            emotion = dialogue.get('emotion', '')
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            situations = dialogue.get('situation', [])
            for situation in situations:
                situation_counts[situation] = situation_counts.get(situation, 0) + 1

        # 이미지 카테고리 분포
        category_counts = {}
        for img in self.image_data:
            category = img.get('category', '')
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1

        return {
            'total_dialogues': len(self.dialogue_data),
            'total_images': len(self.image_data),
            'emotion_distribution': emotion_counts,
            'situation_distribution': situation_counts,
            'image_category_distribution': category_counts
        }

    def print_statistics(self):
        """데이터 통계 정보 출력"""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("📊 데이터 통계 정보")
        print("=" * 50)

        print(f"총 대화 수: {stats.get('total_dialogues', 0)}개")
        print(f"총 이미지 수: {stats.get('total_images', 0)}개")

        print("\n😊 감정 분포:")
        for emotion, count in stats.get('emotion_distribution', {}).items():
            print(f"  {emotion}: {count}개")

        print("\n🏢 상황 분포:")
        for situation, count in stats.get('situation_distribution', {}).items():
            print(f"  {situation}: {count}개")

        print("\n🖼️ 이미지 카테고리 분포:")
        for category, count in stats.get('image_category_distribution', {}).items():
            print(f"  {category}: {count}개")


# 테스트용 메인 함수
if __name__ == "__main__":
    # 데이터 로더 테스트
    loader = DialogueDataLoader()

    # 통계 정보 출력
    loader.print_statistics()

    # 학습 쌍 생성 테스트
    training_pairs = loader.get_training_pairs()

    if training_pairs:
        print(f"\n[테스트] 첫 번째 학습 쌍:")
        first_pair = training_pairs[0]
        for key, value in first_pair.items():
            print(f"  {key}: {value}")

    # 데이터 분할 테스트
    train_data, test_data = loader.split_data(test_size=0.2)