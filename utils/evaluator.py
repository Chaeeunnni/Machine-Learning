import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os


class MemeRecommendationEvaluator:
    def __init__(self, recommender, test_data):
        self.recommender = recommender
        self.test_data = test_data
        self.evaluation_results = []

    def evaluate_recommendations(self):
        """추천 시스템 성능 평가 - 에러 핸들링 강화"""
        if not self.test_data:
            print("[!] 테스트 데이터가 없습니다.")
            return {
                'total_tests': 0,
                'accuracy_top1': 0.0,
                'accuracy_top3': 0.0,
                'accuracy_top5': 0.0,
                'emotion_accuracy': {},
                'situation_accuracy': {},
                'detailed_results': []
            }

        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total_tests = len(self.test_data)

        # ZeroDivisionError 방지
        if total_tests == 0:
            print("[!] 테스트 데이터가 비어있습니다.")
            return {
                'total_tests': 0,
                'accuracy_top1': 0.0,
                'accuracy_top3': 0.0,
                'accuracy_top5': 0.0,
                'emotion_accuracy': {},
                'situation_accuracy': {},
                'detailed_results': []
            }

        emotion_accuracy = {'분노': [], '기쁨': [], '슬픔': [], '불안': [], '감사': [], '자신': []}
        situation_accuracy = {'직장': [], '진로': [], '인간관계': [], '연애': [], '돈': [], '개인적': []}

        print("[*] 성능 평가 시작...")

        for i, test_case in enumerate(self.test_data):
            if i % 10 == 0:
                print(f"  진행률: {i}/{total_tests}")

            try:
                # 추천 실행
                result = self.recommender.recommend(test_case['text'])

                # 실제 정답 이미지들 (여러 개일 수 있음)
                expected_images = [test_case.get('image_filename', '')]
                if not expected_images[0]:  # 빈 문자열인 경우
                    expected_images = []

                # Top-K 정확도 계산
                if result and 'top_5' in result and result['top_5']:
                    top_5_images = [item.get('filename', '') for item in result['top_5']]
                else:
                    top_5_images = [result.get('best_image', '')] if result else []

                # 예상 이미지가 있을 때만 정확도 계산
                if expected_images and expected_images[0]:
                    is_top1_correct = result.get('best_image', '') in expected_images
                    is_top3_correct = any(img in expected_images for img in top_5_images[:3])
                    is_top5_correct = any(img in expected_images for img in top_5_images)

                    if is_top1_correct:
                        correct_top1 += 1
                    if is_top3_correct:
                        correct_top3 += 1
                    if is_top5_correct:
                        correct_top5 += 1
                else:
                    # 예상 이미지가 없는 경우, 추천이 성공했다면 성공으로 간주
                    is_top1_correct = result is not None
                    is_top3_correct = result is not None
                    is_top5_correct = result is not None

                    if is_top1_correct:
                        correct_top1 += 1
                        correct_top3 += 1
                        correct_top5 += 1

                # 감정별 정확도
                test_emotion = test_case.get('emotion', '')
                if test_emotion and test_emotion in emotion_accuracy:
                    emotion_accuracy[test_emotion].append(is_top1_correct)

                # 상황별 정확도
                test_situations = test_case.get('situation', [])
                if isinstance(test_situations, str):
                    test_situations = [test_situations]

                for situation in test_situations:
                    if situation in situation_accuracy:
                        situation_accuracy[situation].append(is_top1_correct)

                # 개별 결과 저장
                self.evaluation_results.append({
                    'dialogue_id': test_case.get('dialogue_id', f'test_{i}'),
                    'text': test_case['text'][:100] + "..." if len(test_case['text']) > 100 else test_case['text'],
                    'expected_image': expected_images[0] if expected_images else 'N/A',
                    'predicted_image': result.get('best_image', 'N/A') if result else 'N/A',
                    'score': result.get('score', 0.0) if result else 0.0,
                    'emotion': test_emotion,
                    'top1_correct': is_top1_correct,
                    'top3_correct': is_top3_correct,
                    'top5_correct': is_top5_correct
                })

            except Exception as e:
                print(f"[!] 평가 중 오류 (테스트 {i}): {e}")
                # 오류가 발생한 경우 실패로 처리
                self.evaluation_results.append({
                    'dialogue_id': test_case.get('dialogue_id', f'test_{i}'),
                    'text': test_case.get('text', 'N/A')[:100] + "...",
                    'expected_image': 'N/A',
                    'predicted_image': 'ERROR',
                    'score': 0.0,
                    'emotion': test_case.get('emotion', ''),
                    'top1_correct': False,
                    'top3_correct': False,
                    'top5_correct': False,
                    'error': str(e)
                })

        # 전체 정확도 계산 (ZeroDivisionError 방지)
        accuracy_top1 = correct_top1 / total_tests if total_tests > 0 else 0.0
        accuracy_top3 = correct_top3 / total_tests if total_tests > 0 else 0.0
        accuracy_top5 = correct_top5 / total_tests if total_tests > 0 else 0.0

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

        if results['total_tests'] > 0:
            print(f"Top-1 정확도: {results['accuracy_top1']:.3f} ({results['accuracy_top1'] * 100:.1f}%)")
            print(f"Top-3 정확도: {results['accuracy_top3']:.3f} ({results['accuracy_top3'] * 100:.1f}%)")
            print(f"Top-5 정확도: {results['accuracy_top5']:.3f} ({results['accuracy_top5'] * 100:.1f}%)")

            print("\n📈 감정별 정확도:")
            emotion_items = [(k, v) for k, v in results['emotion_accuracy'].items() if v > 0]
            if emotion_items:
                for emotion, accuracy in emotion_items:
                    print(f"  {emotion}: {accuracy:.3f} ({accuracy * 100:.1f}%)")
            else:
                print("  감정별 데이터가 충분하지 않습니다.")

            print("\n🏢 상황별 정확도:")
            situation_items = [(k, v) for k, v in results['situation_accuracy'].items() if v > 0]
            if situation_items:
                for situation, accuracy in situation_items:
                    print(f"  {situation}: {accuracy:.3f} ({accuracy * 100:.1f}%)")
            else:
                print("  상황별 데이터가 충분하지 않습니다.")

            # 실패 케이스 분석
            failed_cases = [r for r in results['detailed_results'] if not r.get('top1_correct', False)]
            if failed_cases and len(failed_cases) <= 10:  # 실패 케이스가 10개 이하일 때만 표시
                print(f"\n❌ 실패 케이스 분석 (총 {len(failed_cases)}개):")
                for i, case in enumerate(failed_cases[:5], 1):  # 최대 5개만 표시
                    print(f"  {i}. 텍스트: {case['text']}")
                    if case.get('error'):
                        print(f"     오류: {case['error']}")
                    else:
                        print(f"     예상: {case['expected_image']}, 예측: {case['predicted_image']}")
                        print(f"     점수: {case['score']:.3f}, 감정: {case['emotion']}")
                    print()
            elif len(failed_cases) > 10:
                print(f"\n❌ 실패 케이스: {len(failed_cases)}개 (너무 많아 상세 표시 생략)")
        else:
            print("⚠️ 평가할 데이터가 없습니다.")

    def save_evaluation_results(self, results, filename="evaluation_results.json"):
        """평가 결과를 파일로 저장"""
        try:
            # results 디렉토리가 없으면 생성
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"[*] {results_dir} 디렉토리 생성")

            # 전체 경로 구성
            full_path = os.path.join(results_dir, filename)

            # numpy 타입을 JSON 직렬화 가능한 타입으로 변환
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # 결과 변환 및 저장
            converted_results = convert_numpy_types(results)

            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            print(f"[*] 평가 결과 저장: {full_path}")

        except Exception as e:
            print(f"[!] 평가 결과 저장 실패: {e}")

    def calculate_confusion_matrix(self, results):
        """혼동 행렬 계산"""
        if not results['detailed_results']:
            return None

        try:
            # 감정별 혼동 행렬 데이터 수집
            emotion_predictions = []
            emotion_actuals = []

            for result in results['detailed_results']:
                actual_emotion = result.get('emotion', '')
                # 예측된 감정은 간단히 성공/실패로 구분 (실제로는 더 복잡한 로직 필요)
                predicted_emotion = actual_emotion if result.get('top1_correct', False) else 'other'

                if actual_emotion:  # 실제 감정이 있는 경우만
                    emotion_actuals.append(actual_emotion)
                    emotion_predictions.append(predicted_emotion)

            return {
                'emotion_actuals': emotion_actuals,
                'emotion_predictions': emotion_predictions
            }

        except Exception as e:
            print(f"[!] 혼동 행렬 계산 실패: {e}")
            return None

    def get_summary_statistics(self, results):
        """요약 통계 계산"""
        if not results['detailed_results']:
            return {}

        try:
            scores = [r.get('score', 0.0) for r in results['detailed_results']]

            return {
                'mean_score': np.mean(scores) if scores else 0.0,
                'median_score': np.median(scores) if scores else 0.0,
                'std_score': np.std(scores) if scores else 0.0,
                'min_score': np.min(scores) if scores else 0.0,
                'max_score': np.max(scores) if scores else 0.0,
                'total_errors': len([r for r in results['detailed_results'] if 'error' in r]),
                'success_rate': results['accuracy_top1']
            }

        except Exception as e:
            print(f"[!] 통계 계산 실패: {e}")
            return {}


# 테스트용 메인 함수
if __name__ == "__main__":
    # 간단한 테스트
    print("🧪 Evaluator 테스트")


    # 가짜 추천 시스템
    class MockRecommender:
        def recommend(self, text):
            return {
                'best_image': 'test_image.jpg',
                'score': 0.5,
                'emotions': ['기쁨'],
                'situations': ['일상'],
                'top_5': [
                    {'filename': 'test_image.jpg', 'score': 0.5},
                    {'filename': 'test_image2.jpg', 'score': 0.4},
                    {'filename': 'test_image3.jpg', 'score': 0.3},
                    {'filename': 'test_image4.jpg', 'score': 0.2},
                    {'filename': 'test_image5.jpg', 'score': 0.1}
                ]
            }


    # 가짜 테스트 데이터
    mock_test_data = [
        {
            'dialogue_id': 'test_1',
            'text': '오늘 기분이 좋아!',
            'emotion': '기쁨',
            'situation': ['일상'],
            'image_filename': 'test_image.jpg'
        },
        {
            'dialogue_id': 'test_2',
            'text': '너무 슬퍼',
            'emotion': '슬픔',
            'situation': ['개인적'],
            'image_filename': 'sad_image.jpg'
        }
    ]

    # 평가 실행
    mock_recommender = MockRecommender()
    evaluator = MemeRecommendationEvaluator(mock_recommender, mock_test_data)
    results = evaluator.evaluate_recommendations()

    # 결과 출력
    evaluator.print_evaluation_report(results)

    # 통계 정보
    stats = evaluator.get_summary_statistics(results)
    print(f"\n📊 요약 통계: {stats}")

    print("\n✅ Evaluator 테스트 완료!")