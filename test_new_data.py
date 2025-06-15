import os
import warnings
from transformers import logging

# 경고 메시지 비활성화
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from models.hybrid_encoder import HybridEncoder
from models.matching import HybridMatcher
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from PIL import Image as PILImage


# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 자동 설정"""
    if platform.system() == 'Windows':
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            print("[*] 한글 폰트 설정: Malgun Gothic")
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("[*] 기본 폰트 사용")
    else:
        try:
            import matplotlib.font_manager as fm
            font_list = [f.name for f in fm.fontManager.ttflist]

            korean_fonts = ['NanumGothic', 'AppleGothic', 'Noto Sans CJK KR']
            selected_font = None

            for font in korean_fonts:
                if font in font_list:
                    selected_font = font
                    break

            if selected_font:
                plt.rcParams['font.family'] = selected_font
                print(f"[*] 한글 폰트 설정: {selected_font}")
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                print("[*] 기본 폰트 사용")
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("[*] 기본 폰트 사용")

    plt.rcParams['axes.unicode_minus'] = False


# 폰트 설정 실행
setup_korean_font()


class InteractiveMemeRecommender:
    def __init__(self, image_metadata="data/enhanced_image_metadata.json"):
        print("[*] 🎯 대화형 짤 추천 시스템 초기화 중...")

        # 하이브리드 모델 초기화
        self.hybrid_encoder = HybridEncoder()
        self.matcher = HybridMatcher(image_metadata)

        # 이미지 로딩
        self.load_images_from_metadata(image_metadata)
        self.precompute_image_embeddings()

        # 추천 기록 저장용
        self.recommendation_history = []

        print("✅ 시스템 준비 완료! 이제 대화를 입력해보세요.")

    def load_images_from_metadata(self, metadata_file):
        """메타데이터에서 이미지 로딩"""
        self.image_files = []

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                for item in metadata:
                    if item.get('processing_status') == 'success':
                        img_path = item.get('processed_path') or item.get('filepath')
                        if img_path and os.path.exists(img_path):
                            self.image_files.append(img_path)
                        else:
                            relative_path = f"data/images/{item.get('filepath', '')}"
                            if os.path.exists(relative_path):
                                self.image_files.append(relative_path)

                print(f"[*] 메타데이터에서 {len(self.image_files)}개 이미지 로드")

            except Exception as e:
                print(f"[!] 메타데이터 로드 실패: {e}")
                self.load_images_fallback()
        else:
            print("[!] 메타데이터 파일이 없음. 폴백 모드로 전환...")
            self.load_images_fallback()

    def load_images_fallback(self):
        """폴백: 디렉토리에서 직접 이미지 로드"""
        image_patterns = [
            "data/images/**/*.jpg",
            "data/images/**/*.png",
            "data/images/**/*.jpeg",
            "data/images/**/*.webp"
        ]

        self.image_files = []
        for pattern in image_patterns:
            found_files = glob.glob(pattern, recursive=True)
            self.image_files.extend(found_files)

        print(f"[*] 폴백 모드로 {len(self.image_files)}개 이미지 로드")

    def precompute_image_embeddings(self):
        """이미지 임베딩 미리 계산"""
        if not self.image_files:
            print("[!] 이미지 파일이 없습니다.")
            return

        print("[*] 이미지 임베딩 생성 중...")
        self.image_embeddings = []

        for i, img_path in enumerate(self.image_files):
            if i % 20 == 0:
                print(f"  📊 진행률: {i}/{len(self.image_files)}")

            try:
                emb = self.hybrid_encoder.get_image_embedding(img_path)
                self.image_embeddings.append(emb)
            except Exception as e:
                print(f"[!] 이미지 임베딩 실패: {img_path}")
                self.image_embeddings.append(np.zeros(512))

        self.image_embeddings = np.array(self.image_embeddings)
        print(f"[*] ✅ 이미지 임베딩 생성 완료! (총 {len(self.image_embeddings)}개)")

    def recommend_meme(self, dialogue_text, show_details=True, show_image=True):
        """대화 텍스트에 대한 짤 추천 - 항상 최적의 추천 제공"""
        if not hasattr(self, 'image_embeddings') or len(self.image_embeddings) == 0:
            print("❌ 이미지 임베딩이 없습니다.")
            return None

        print(f"\n🔍 분석 중: '{dialogue_text}'")

        try:
            # 하이브리드 매칭 수행
            best_idx, score, top_5, emotions, situations = self.matcher.find_best_match_hybrid(
                self.hybrid_encoder, dialogue_text, self.image_embeddings, self.image_files
            )

            recommended_image_path = self.image_files[best_idx]
            recommended_image_name = os.path.basename(recommended_image_path)

            # 결과 구성
            result = {
                'input_text': dialogue_text,
                'recommended_image': recommended_image_name,
                'image_path': recommended_image_path,
                'confidence_score': score,
                'detected_emotions': emotions,
                'detected_situations': situations,
                'alternatives': top_5[:3],  # 상위 3개 대안
                'recommendation_quality': self.evaluate_recommendation_quality(score)
            }

            # 추천 기록 저장
            self.recommendation_history.append(result)

            # 결과 출력
            self.display_recommendation(result, show_details, show_image)

            return result

        except Exception as e:
            print(f"❌ 추천 실패: {e}")
            return None

    def evaluate_recommendation_quality(self, score):
        """추천 품질 평가"""
        if score >= 0.4:
            return "🔥 매우 적합"
        elif score >= 0.3:
            return "👍 적합"
        elif score >= 0.2:
            return "⭐ 보통"
        else:
            return "🤔 참고용"

    def display_recommendation(self, result, show_details=True, show_image=True):
        """추천 결과 예쁘게 표시"""
        print("\n" + "=" * 60)
        print("🎯 짤 추천 결과")
        print("=" * 60)

        print(f"💬 입력: {result['input_text']}")
        print(f"🖼️  추천 이미지: {result['recommended_image']}")
        print(f"📊 신뢰도: {result['confidence_score']:.4f} ({result['recommendation_quality']})")

        if show_details:
            if result['detected_emotions']:
                emotions_str = ', '.join(result['detected_emotions'])
                print(f"😊 감지된 감정: {emotions_str}")
            else:
                print("😊 감지된 감정: 일반적인 상황")

            if result['detected_situations']:
                situations_str = ', '.join(result['detected_situations'])
                print(f"🏢 감지된 상황: {situations_str}")
            else:
                print("🏢 감지된 상황: 일상 대화")

            print(f"\n🏆 다른 추천 후보:")
            for i, alt in enumerate(result['alternatives'], 1):
                category = alt.get('category', '알 수 없음')
                subcategory = alt.get('subcategory', '알 수 없음')
                print(f"   {i}. {alt['filename']} - {category}/{subcategory} ({alt['score']:.4f})")

        # 이미지 표시
        if show_image:
            self.display_image(result['image_path'])

    def display_image(self, image_path):
        """이미지 시각적으로 표시"""
        try:
            plt.figure(figsize=(8, 6))
            img = PILImage.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"추천된 짤: {os.path.basename(image_path)}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[!] 이미지 표시 실패: {e}")
            print(f"📁 이미지 경로: {image_path}")

    def interactive_session(self):
        """대화형 세션 시작"""
        print("\n🎉 대화형 짤 추천 시스템에 오신 것을 환영합니다!")
        print("💡 사용법:")
        print("   - 대화나 감정을 자유롭게 입력하세요")
        print("   - 'quit', 'exit', '종료'를 입력하면 종료됩니다")
        print("   - 'history'를 입력하면 추천 기록을 볼 수 있습니다")
        print("   - 'stats'를 입력하면 통계를 볼 수 있습니다")
        print("-" * 60)

        session_count = 0

        while True:
            try:
                user_input = input(f"\n[{session_count + 1}] 💬 대화를 입력하세요: ").strip()

                if not user_input:
                    print("❌ 빈 텍스트입니다. 다시 입력해주세요.")
                    continue

                # 종료 명령어
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 시스템을 종료합니다. 감사합니다!")
                    break

                # 기록 보기
                elif user_input.lower() in ['history', '기록']:
                    self.show_history()
                    continue

                # 통계 보기
                elif user_input.lower() in ['stats', '통계']:
                    self.show_statistics()
                    continue

                # 도움말
                elif user_input.lower() in ['help', '도움말']:
                    self.show_help()
                    continue

                # 짤 추천 실행
                result = self.recommend_meme(user_input, show_details=True, show_image=True)

                if result:
                    session_count += 1

                    # 사용자 피드백 받기
                    feedback = input("\n👍 이 추천이 도움이 되었나요? (y/n/enter): ").strip().lower()
                    if feedback == 'y':
                        result['user_feedback'] = 'positive'
                        print("😊 피드백 감사합니다!")
                    elif feedback == 'n':
                        result['user_feedback'] = 'negative'
                        print("😅 다음에는 더 좋은 추천을 드릴게요!")
                    else:
                        result['user_feedback'] = 'neutral'

            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                continue

        # 세션 종료 시 통계 표시
        if session_count > 0:
            print(f"\n📊 세션 통계: 총 {session_count}개의 추천을 제공했습니다.")
            self.show_session_summary()

    def show_history(self):
        """추천 기록 표시"""
        if not self.recommendation_history:
            print("📝 아직 추천 기록이 없습니다.")
            return

        print(f"\n📚 추천 기록 (총 {len(self.recommendation_history)}개)")
        print("-" * 60)

        for i, record in enumerate(self.recommendation_history[-10:], 1):  # 최근 10개만
            feedback = record.get('user_feedback', 'no feedback')
            feedback_emoji = {'positive': '👍', 'negative': '👎', 'neutral': '😐', 'no feedback': '❓'}

            print(f"{i}. 💬 '{record['input_text'][:30]}...'")
            print(f"   🖼️  {record['recommended_image']} ({record['confidence_score']:.3f}) {feedback_emoji[feedback]}")

    def show_statistics(self):
        """통계 정보 표시"""
        if not self.recommendation_history:
            print("📊 아직 통계 데이터가 없습니다.")
            return

        total_recommendations = len(self.recommendation_history)
        avg_score = np.mean([r['confidence_score'] for r in self.recommendation_history])

        # 피드백 통계
        feedbacks = [r.get('user_feedback', 'no feedback') for r in self.recommendation_history]
        positive_feedback = feedbacks.count('positive')
        negative_feedback = feedbacks.count('negative')

        # 감정 통계
        all_emotions = []
        for r in self.recommendation_history:
            all_emotions.extend(r['detected_emotions'])

        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print(f"\n📊 세션 통계")
        print("-" * 40)
        print(f"총 추천 수: {total_recommendations}개")
        print(f"평균 신뢰도: {avg_score:.3f}")
        print(f"긍정 피드백: {positive_feedback}개 ({positive_feedback / total_recommendations * 100:.1f}%)")
        print(f"부정 피드백: {negative_feedback}개 ({negative_feedback / total_recommendations * 100:.1f}%)")

        if emotion_counts:
            print(f"\n가장 많이 감지된 감정:")
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_emotions[:3]:
                print(f"  {emotion}: {count}회")

    def show_help(self):
        """도움말 표시"""
        print("\n❓ 도움말")
        print("-" * 40)
        print("🔹 자유롭게 감정이나 상황을 텍스트로 입력하세요")
        print("🔹 예시:")
        print("   - '오늘 정말 기분이 좋아!'")
        print("   - '상사가 너무 화나게 해서 스트레스 받아'")
        print("   - '친구랑 헤어져서 슬퍼'")
        print("   - '시험 합격해서 기뻐!'")
        print("🔹 명령어:")
        print("   - 'history' 또는 '기록': 추천 기록 보기")
        print("   - 'stats' 또는 '통계': 통계 정보 보기")
        print("   - 'quit' 또는 '종료': 시스템 종료")

    def show_session_summary(self):
        """세션 요약 표시"""
        if not self.recommendation_history:
            return

        # 통계 계산
        total = len(self.recommendation_history)
        avg_score = np.mean([r['confidence_score'] for r in self.recommendation_history])

        feedbacks = [r.get('user_feedback', 'no feedback') for r in self.recommendation_history]
        positive_rate = feedbacks.count('positive') / total * 100 if total > 0 else 0

        print(f"\n📈 세션 요약")
        print(f"   총 추천: {total}개")
        print(f"   평균 신뢰도: {avg_score:.3f}")
        print(f"   만족도: {positive_rate:.1f}%")

        # 시각화
        if total >= 3:
            self.visualize_session_stats()

    def visualize_session_stats(self):
        """세션 통계 시각화"""
        if len(self.recommendation_history) < 3:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('세션 통계 대시보드', fontsize=16, fontweight='bold')

        # 1. 신뢰도 추이
        scores = [r['confidence_score'] for r in self.recommendation_history]
        axes[0, 0].plot(range(1, len(scores) + 1), scores, 'b-o', linewidth=2)
        axes[0, 0].set_title('추천 신뢰도 추이')
        axes[0, 0].set_xlabel('추천 순서')
        axes[0, 0].set_ylabel('신뢰도 점수')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 감정 분포
        all_emotions = []
        for r in self.recommendation_history:
            all_emotions.extend(r['detected_emotions'])

        if all_emotions:
            emotion_counts = {}
            for emotion in all_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())

            axes[0, 1].bar(emotions, counts, color=plt.cm.Set3(np.linspace(0, 1, len(emotions))))
            axes[0, 1].set_title('감지된 감정 분포')
            axes[0, 1].set_ylabel('빈도')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, '감지된 감정 없음', ha='center', va='center')
            axes[0, 1].set_title('감지된 감정 분포')

        # 3. 피드백 분포
        feedbacks = [r.get('user_feedback', 'no feedback') for r in self.recommendation_history]
        feedback_counts = {
            '긍정': feedbacks.count('positive'),
            '부정': feedbacks.count('negative'),
            '보통': feedbacks.count('neutral'),
            '없음': feedbacks.count('no feedback')
        }

        colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E']
        axes[1, 0].pie(feedback_counts.values(), labels=feedback_counts.keys(),
                       colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('사용자 피드백 분포')

        # 4. 신뢰도 분포
        axes[1, 1].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('신뢰도 점수 분포')
        axes[1, 1].set_xlabel('신뢰도 점수')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].axvline(np.mean(scores), color='red', linestyle='--',
                           label=f'평균: {np.mean(scores):.3f}')
        axes[1, 1].legend()

        plt.tight_layout()

        # 저장
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/session_stats.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("[*] 세션 통계가 'results/session_stats.png'에 저장되었습니다.")

    def batch_test(self, test_texts):
        """여러 텍스트를 한번에 테스트"""
        print(f"\n🧪 배치 테스트 시작 (총 {len(test_texts)}개)")
        print("=" * 60)

        results = []
        for i, text in enumerate(test_texts, 1):
            print(f"\n[{i}/{len(test_texts)}]")
            result = self.recommend_meme(text, show_details=False, show_image=False)
            if result:
                results.append(result)

        # 배치 결과 요약
        if results:
            avg_score = np.mean([r['confidence_score'] for r in results])
            print(f"\n📊 배치 테스트 완료!")
            print(f"   평균 신뢰도: {avg_score:.3f}")
            print(f"   총 추천: {len(results)}개")

        return results


def main():
    """메인 함수 - 대화형 세션 시작"""
    try:
        # 시스템 초기화
        recommender = InteractiveMemeRecommender()

        # 대화형 세션 시작
        recommender.interactive_session()

    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")


def demo_batch_test():
    """데모용 배치 테스트"""
    recommender = InteractiveMemeRecommender()

    demo_texts = [
        "오늘 정말 기분이 좋아! 승진 소식을 들었어!",
        "상사가 계속 야근을 시켜서 정말 지쳤어.",
        "친구가 도움을 줘서 정말 감사해.",
        "연인과 헤어져서 너무 슬퍼.",
        "새로운 기술을 배우는 게 재미있어!",
        "시험에서 떨어져서 실망스러워.",
        "가족들과 함께 시간을 보내서 행복해.",
        "일이 너무 많아서 스트레스 받아."
    ]

    return recommender.batch_test(demo_texts)


if __name__ == "__main__":
    print("🎯 짤 추천 시스템")
    print("1. 대화형 모드 (기본)")
    print("2. 데모 배치 테스트")

    choice = input("선택하세요 (1/2, 기본값: 1): ").strip()

    if choice == "2":
        demo_batch_test()
    else:
        main()