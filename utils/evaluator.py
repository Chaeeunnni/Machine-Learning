import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


class MemeRecommendationEvaluator:
    def __init__(self, recommender, test_data):
        self.recommender = recommender
        self.test_data = test_data
        self.evaluation_results = []

    def evaluate_recommendations(self):
        """추천 시스템 성능 평가"""
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total_tests = len(self.test_data)

        emotion_accuracy = {'분노': [], '기쁨': [], '슬픔': [], '불안': []}
        situation_accuracy = {'직장': [], '진로': [], '인간관계': []}

        print("[*] 성능 평가 시작...")

        for i, test_case in enumerate(self.test_data):
            if i % 10 == 0:
                print(f"  진행률: {i}/{total_tests}")

            # 추천 실행
            result = self.recommender.recommend(test_case['text'])

            # 실제 정답 이미지들
            expected_images = [test_case['image_filename']]

            # Top-K 정확도 계산
            top_5_images = [item['filename'] for item in result['top_5']]

            is_top1_correct = result['best_image'] in expected_images
            is_top3_correct = any(img in expected_images for img in top_5_images[:3])
            is_top5_correct = any(img in expected_images for img in top_5_images)

            if is_top1_correct:
                correct_top1 += 1
            if is_top3_correct:
                correct_top3 += 1
            if is_top5_correct:
                correct_top5 += 1

            # 감정별 정확도
            emotion = test_case['emotion']
            if emotion in emotion_accuracy:
                emotion_accuracy[emotion].append(is_top1_correct)

            # 상황별 정확도
            for situation in test_case['situation']:
                if situation in situation_accuracy:
                    situation_accuracy[situation].append(is_top1_correct)

            # 개별 결과 저장
            self.evaluation_results.append({
                'dialogue_id': test_case['dialogue_id'],
                'text': test_case['text'][:100] + "...",
                'expected_image': test_case['image_filename'],
                'predicted_image': result['best_image'],
                'score': result['score'],
                'emotion': emotion,
                'top1_correct': is_top1_correct,
                'top3_correct': is_top3_correct,
                'top5_correct': is_top5_correct
            })

        # 전체 정확도 계산
        accuracy_top1 = correct_top1 / total_tests
        accuracy_top3 = correct_top3 / total_tests
        accuracy_top5 = correct_top5 / total_tests

        # 감정별 정확도 계산
        emotion_results = {}
        for emotion, results in emotion_accuracy.items():
            if results:
                emotion_results[emotion] = sum(results) / len(results)
            else:
                emotion_results[emotion] = 0.0

        # 상황별 정확도 계산
        situation_results = {}
        for situation, results in situation_accuracy.items():
            if results:
                situation_results[situation] = sum(results) / len(results)
            else:
                situation_results[situation] = 0.0

        return {
            'total_tests': total_tests,
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'accuracy_top5': accuracy_top5,
            'emotion_accuracy': emotion_results,
            'situation_accuracy': situation_results,
            'detailed_results': self.evaluation_results
        }

    def print_evaluation_report(self, results):
        """평가 결과 출력"""
        print("\n" + "=" * 60)
        print("📊 성능 평가 결과")
        print("=" * 60)

        print(f"총 테스트 케이스: {results['total_tests']}개")
        print(f"Top-1 정확도: {results['accuracy_top1']:.3f} ({results['accuracy_top1'] * 100:.1f}%)")
        print(f"Top-3 정확도: {results['accuracy_top3']:.3f} ({results['accuracy_top3'] * 100:.1f}%)")
        print(f"Top-5 정확도: {results['accuracy_top5']:.3f} ({results['accuracy_top5'] * 100:.1f}%)")

        print("\n📈 감정별 정확도:")
        for emotion, accuracy in results['emotion_accuracy'].items():
            print(f"  {emotion}: {accuracy:.3f} ({accuracy * 100:.1f}%)")

        print("\n🏢 상황별 정확도:")
        for situation, accuracy in results['situation_accuracy'].items():
            print(f"  {situation}: {accuracy:.3f} ({accuracy * 100:.1f}%)")

        # 실패 케이스 분석
        failed_cases = [r for r in results['detailed_results'] if not r['top1_correct']]
        if failed_cases:
            print(f"\n❌ 실패 케이스 분석 (상위 5개):")
            for case in failed_cases[:5]:
                print(f"  텍스트: {case['text']}")
                print(f"  예상: {case['expected_image']}, 예측: {case['predicted_image']}")
                print(f"  점수: {case['score']:.3f}, 감정: {case['emotion']}")
                print()

    def save_evaluation_results(self, results, filename="evaluation_results.json"):
        """평가 결과를 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[*] 평가 결과 저장: {filename}")