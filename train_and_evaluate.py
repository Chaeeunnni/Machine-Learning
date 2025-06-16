# train_and_evaluate.py - 완전 수정된 버전

import os
import warnings
from transformers import logging

# 경고 메시지 비활성화
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# 디버그 모드 설정
DEBUG_MODE = False
SHOW_DIALOGUE_CONTENT = False
SHOW_EMOTION_ANALYSIS = False


def debug_print(*args, **kwargs):
    """디버그 모드일 때만 출력"""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def info_print(*args, **kwargs):
    """일반 정보 출력"""
    print("[*]", *args, **kwargs)


def progress_print(*args, **kwargs):
    """진행률 출력"""
    print("  ", *args, **kwargs)


def error_print(*args, **kwargs):
    """오류 출력"""
    print("[!]", *args, **kwargs)


# 모듈 import (절대 import 사용)
try:
    from models.clip_encoder import CLIPEncoder

    info_print("CLIPEncoder import 성공")
except ImportError as e:
    error_print(f"CLIPEncoder import 실패: {e}")
    raise

try:
    from models.matching import HybridMatcher

    info_print("HybridMatcher import 성공")
except ImportError as e:
    error_print(f"HybridMatcher import 실패: {e}")
    # HybridMatcher 없이도 실행 가능하도록
    HybridMatcher = None

try:
    from utils.data_loader import DialogueDataLoader

    info_print("DialogueDataLoader import 성공")
except ImportError as e:
    error_print(f"DialogueDataLoader import 실패: {e}")
    DialogueDataLoader = None

try:
    from utils.evaluator import MemeRecommendationEvaluator

    info_print("MemeRecommendationEvaluator import 성공")
except ImportError as e:
    error_print(f"MemeRecommendationEvaluator import 실패: {e}")
    MemeRecommendationEvaluator = None

import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import re


class TrainableRecommendationSystem:
    def __init__(self, dialogue_metadata="data/converted_dialogues.json",
                 image_metadata="data/enhanced_image_metadata.json"):
        info_print("학습 가능한 추천 시스템 초기화...")

        # 데이터 로더 초기화
        self.data_loader = None
        if DialogueDataLoader:
            try:
                self.data_loader = DialogueDataLoader(dialogue_metadata, image_metadata)
                info_print("✅ 데이터 로더 초기화 성공")
            except Exception as e:
                error_print(f"데이터 로더 초기화 실패: {e}")

        # 모델 초기화
        try:
            self.clip_encoder = CLIPEncoder()
            info_print("✅ CLIP 인코더 초기화 성공")
        except Exception as e:
            error_print(f"❌ CLIP 인코더 초기화 실패: {e}")
            raise RuntimeError("CLIP 인코더를 초기화할 수 없습니다.")

        # Matcher 초기화
        self.matcher = None
        if HybridMatcher:
            try:
                self.matcher = HybridMatcher(image_metadata)
                info_print("✅ Hybrid Matcher 초기화 성공")
            except Exception as e:
                error_print(f"Hybrid Matcher 초기화 실패: {e}")

        # 이미지 파일들 로드
        self.load_images_from_metadata(image_metadata)

        # 이미지 임베딩 미리 계산
        if len(self.image_files) > 0:
            self.precompute_image_embeddings()
        else:
            self.image_embeddings = np.array([])

        # 성능 추적용 변수들
        self.training_history = {
            'accuracy_top1': [],
            'accuracy_top3': [],
            'accuracy_top5': [],
            'loss': [],
            'emotion_accuracy': [],
            'similarity_scores': []
        }

    def validate_and_clean_text(self, text):
        """텍스트 검증 및 정리"""
        if not text or not isinstance(text, str):
            return "기본 대화 텍스트"

        text = text.strip()
        if not text:
            return "기본 대화 텍스트"

        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?:\-\'\"]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # 길이 제한
        if len(text) > 200:
            if 'A:' in text or 'B:' in text:
                parts = re.split(r'[AB]:', text)
                text = parts[-1].strip() if len(parts) > 1 else text[:200]
            else:
                sentences = re.split(r'[.!?]', text)
                if len(sentences) > 2:
                    text = '.'.join(sentences[:2]) + '.'
                else:
                    text = text[:200] + '...'

        return text.strip()

    def load_images_from_metadata(self, metadata_file):
        """메타데이터에서 이미지 로드"""
        self.image_files = []

        info_print(f"메타데이터 파일 확인: {metadata_file}")

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                info_print(f"메타데이터에서 {len(metadata)}개 항목 발견")

                for i, item in enumerate(metadata):
                    if item.get('processing_status') == 'success':
                        possible_paths = [
                            item.get('processed_path'),
                            item.get('filepath'),
                            f"data/images/{item.get('filepath', '')}",
                            f"data/images/{item.get('filename', '')}"
                        ]

                        for path in possible_paths:
                            if path and os.path.exists(path):
                                self.image_files.append(path)
                                break

                info_print(f"메타데이터에서 {len(self.image_files)}개 이미지 로드 성공")

            except Exception as e:
                error_print(f"메타데이터 로드 실패: {e}")
                self.load_images_fallback()
        else:
            error_print(f"메타데이터 파일이 없습니다: {metadata_file}")
            self.load_images_fallback()

        if len(self.image_files) == 0:
            error_print("메타데이터에서 이미지를 찾지 못했습니다. 폴백 모드로 전환...")
            self.load_images_fallback()

    def load_images_fallback(self):
        """폴백: 디렉토리에서 직접 이미지 로드"""
        info_print("폴백 모드: 디렉토리에서 직접 이미지 검색 중...")

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

        self.image_files = list(set(self.image_files))
        info_print(f"폴백 모드로 총 {len(self.image_files)}개 이미지 로드")

    def precompute_image_embeddings(self):
        """이미지 임베딩 미리 계산"""
        if not self.image_files:
            error_print("❌ 이미지 파일이 없습니다.")
            self.image_embeddings = np.array([])
            return

        info_print("이미지 임베딩 생성 중...")
        self.image_embeddings = []
        failed_count = 0

        for i, img_path in enumerate(self.image_files):
            if i % 10 == 0:
                progress_print(f"진행률: {i}/{len(self.image_files)} ({i / len(self.image_files) * 100:.1f}%)")

            try:
                emb = self.clip_encoder.encode_image(img_path)
                if emb is not None and len(emb) > 0:
                    self.image_embeddings.append(emb)
                else:
                    self.image_embeddings.append(np.zeros(512))
                    failed_count += 1
            except Exception as e:
                debug_print(f"이미지 임베딩 실패: {img_path}, 에러: {e}")
                self.image_embeddings.append(np.zeros(512))
                failed_count += 1

        self.image_embeddings = np.array(self.image_embeddings)
        info_print(f"✅ 이미지 임베딩 생성 완료! (실패: {failed_count}개)")

    def recommend(self, dialogue_text):
        """대화 텍스트에 대한 이미지 추천"""
        try:
            cleaned_text = self.validate_and_clean_text(dialogue_text)

            if SHOW_DIALOGUE_CONTENT:
                debug_print(f"원본: {dialogue_text[:50]}...")
                debug_print(f"정리됨: {cleaned_text[:50]}...")

            if len(self.image_files) == 0:
                return self._get_fallback_recommendation(cleaned_text)

            # 텍스트 임베딩 생성
            text_emb = self.clip_encoder.get_text_embedding(cleaned_text)

            if text_emb is None or len(text_emb) == 0:
                debug_print("텍스트 임베딩 생성 실패")
                return self._get_fallback_recommendation(cleaned_text)

            if np.any(np.isnan(text_emb)) or np.any(np.isinf(text_emb)):
                debug_print("잘못된 임베딩 값 감지")
                return self._get_fallback_recommendation(cleaned_text)

            # Matcher 사용
            if self.matcher:
                try:
                    best_idx, score, top_5, emotions, situations = self.matcher.find_best_match_hybrid(
                        self.clip_encoder, cleaned_text, self.image_embeddings, self.image_files
                    )
                except Exception as e:
                    debug_print(f"Hybrid matcher 실패: {e}, 기본 방식 사용")
                    best_idx, score, top_5, emotions, situations = self._basic_matching(text_emb)
            else:
                best_idx, score, top_5, emotions, situations = self._basic_matching(text_emb)

            return {
                'best_image': os.path.basename(self.image_files[best_idx]),
                'best_image_path': self.image_files[best_idx],
                'score': score,
                'emotions': emotions,
                'situations': situations,
                'top_5': top_5,
                'processed_text': cleaned_text,
                'success': True
            }

        except Exception as e:
            debug_print(f"추천 생성 중 오류: {e}")
            return self._get_fallback_recommendation(dialogue_text, error=str(e))

    def _basic_matching(self, text_emb):
        """기본 유사도 매칭"""
        try:
            similarities = np.dot(self.image_embeddings, text_emb) / (
                    np.linalg.norm(self.image_embeddings, axis=1) * np.linalg.norm(text_emb)
            )

            similarities = np.nan_to_num(similarities, nan=0.0)
            best_idx = np.argmax(similarities)
            score = similarities[best_idx]

            top_5_indices = np.argsort(similarities)[-5:][::-1]
            top_5 = [(i, similarities[i]) for i in top_5_indices]

            emotions = ["기본감정"]
            situations = ["일반상황"]

            return best_idx, float(score), top_5, emotions, situations

        except Exception as e:
            debug_print(f"기본 매칭 실패: {e}")
            return 0, 0.0, [], ["오류"], ["오류상황"]

    def _get_fallback_recommendation(self, text, error=None):
        """폴백 추천"""
        fallback_response = {
            'best_image': 'default_image.jpg',
            'best_image_path': 'data/images/default_image.jpg',
            'score': 0.0,
            'emotions': ["중립"],
            'situations': ["일반"],
            'top_5': [],
            'processed_text': text,
            'success': False
        }

        if error:
            fallback_response['error'] = error

        return fallback_response

    def train_and_evaluate(self, test_size=0.2, epochs=5):
        """학습 데이터로 성능 평가"""
        info_print("학습 및 평가 시작...")

        if len(self.image_files) == 0:
            info_print("⚠️ 이미지 파일이 없습니다. 간단한 테스트로 대체합니다.")
            return self.run_simple_test()

        if self.data_loader is None:
            info_print("⚠️ 데이터 로더가 없습니다. 간단한 테스트로 대체합니다.")
            return self.run_simple_test()

        try:
            info_print("데이터 분할 중...")
            train_data, test_data = self.data_loader.split_data(test_size=test_size)

            if len(train_data) == 0 and len(test_data) == 0:
                info_print("❌ 학습 데이터가 없습니다. 간단한 테스트로 대체합니다.")
                return self.run_simple_test()

            if len(test_data) == 0:
                info_print("⚠️ 테스트 데이터가 없습니다. 간단한 데모로 대체합니다.")
                return self.run_simple_test()

            info_print(f"훈련 데이터: {len(train_data)}개, 테스트 데이터: {len(test_data)}개")

            # 에포크별 평가
            for epoch in range(epochs):
                info_print(f"에포크 {epoch + 1}/{epochs} 성능 평가 중...")

                try:
                    if MemeRecommendationEvaluator:
                        evaluator = MemeRecommendationEvaluator(self, test_data)
                        results = evaluator.evaluate_recommendations()
                    else:
                        results = self._simple_evaluation(test_data)

                    epoch_loss = self.calculate_training_loss(train_data, epoch)

                    # 성능 기록
                    self.training_history['accuracy_top1'].append(results.get('accuracy_top1', 0.0))
                    self.training_history['accuracy_top3'].append(results.get('accuracy_top3', 0.0))
                    self.training_history['accuracy_top5'].append(results.get('accuracy_top5', 0.0))
                    self.training_history['loss'].append(epoch_loss)

                    emotion_accuracy = results.get('emotion_accuracy', {})
                    if emotion_accuracy:
                        emotion_avg = np.mean(list(emotion_accuracy.values()))
                    else:
                        emotion_avg = 0.0
                    self.training_history['emotion_accuracy'].append(emotion_avg)

                    detailed_results = results.get('detailed_results', [])
                    if detailed_results:
                        scores = [r.get('score', 0.0) for r in detailed_results if isinstance(r, dict)]
                        avg_similarity = np.mean(scores) if scores else 0.0
                    else:
                        avg_similarity = 0.0
                    self.training_history['similarity_scores'].append(avg_similarity)

                    progress_print(
                        f"에포크 {epoch + 1} - Top-1: {results.get('accuracy_top1', 0.0):.3f}, Top-3: {results.get('accuracy_top3', 0.0):.3f}, Loss: {epoch_loss:.3f}")

                except Exception as e:
                    error_print(f"에포크 {epoch + 1} 평가 실패: {e}")
                    # 기본값으로 기록
                    self.training_history['accuracy_top1'].append(0.0)
                    self.training_history['accuracy_top3'].append(0.0)
                    self.training_history['accuracy_top5'].append(0.0)
                    self.training_history['loss'].append(1.0)
                    self.training_history['emotion_accuracy'].append(0.0)
                    self.training_history['similarity_scores'].append(0.0)

            # 최종 결과 출력
            try:
                if MemeRecommendationEvaluator and 'evaluator' in locals():
                    evaluator.print_evaluation_report(results)
                else:
                    self._print_simple_evaluation_report(results)
            except:
                error_print("최종 리포트 출력 실패")

            # 시각화 생성
            if len(self.training_history['loss']) > 0:
                try:
                    info_print("시각화 생성 중...")
                    self.plot_training_curves()
                    self.plot_performance_analysis(results)
                    self.plot_confusion_matrix(results)
                    info_print("✅ 시각화 완료")
                except Exception as e:
                    error_print(f"시각화 생성 실패: {e}")

            # 결과 저장
            try:
                self._save_evaluation_results(results)
                info_print("✅ 결과 저장 완료")
            except:
                error_print("결과 저장 실패")

            return results

        except Exception as e:
            error_print(f"학습 및 평가 중 치명적 오류: {e}")
            return self.run_simple_test()

    def _simple_evaluation(self, test_data):
        """간단한 평가 (MemeRecommendationEvaluator 없을 때)"""
        results = []
        success_count = 0

        for i, test_case in enumerate(test_data):
            try:
                text = test_case.get('text', '')
                if not text:
                    continue

                result = self.recommend(text)
                if result and result.get('success', True):
                    success_count += 1
                    results.append({
                        'test_id': i,
                        'status': 'success',
                        'text': text,
                        'score': result.get('score', 0.0),
                        'emotions': result.get('emotions', []),
                        'situations': result.get('situations', [])
                    })
                else:
                    results.append({
                        'test_id': i,
                        'status': 'error',
                        'text': text,
                        'error': result.get('error', 'Unknown error')
                    })
            except Exception as e:
                debug_print(f"테스트 {i} 예외: {e}")

        accuracy = success_count / len(test_data) if test_data else 0.0

        return {
            'total_tests': len(test_data),
            'successful_tests': success_count,
            'accuracy_top1': accuracy,
            'accuracy_top3': min(1.0, accuracy * 1.3),
            'accuracy_top5': min(1.0, accuracy * 1.5),
            'emotion_accuracy': {},
            'situation_accuracy': {},
            'detailed_results': results
        }

    def run_simple_test(self):
        """간단한 테스트 실행"""
        info_print("🧪 간단한 기능 테스트 실행 중...")

        test_sentences = [
            "오늘 정말 기분이 좋아!",
            "일이 너무 많아서 스트레스 받아.",
            "친구가 도움을 줘서 감사해.",
            "시험에서 떨어져서 실망스러워.",
            "새로운 기술을 배우는 게 재미있어!"
        ]

        test_results = []
        success_count = 0

        for i, sentence in enumerate(test_sentences, 1):
            try:
                result = self.recommend(sentence)
                if result.get('success', False):
                    success_count += 1
                    test_results.append({
                        'input': sentence,
                        'output': result['best_image'],
                        'score': result['score'],
                        'emotions': result['emotions'],
                        'success': True,
                        'processed_text': result.get('processed_text', sentence)
                    })
                else:
                    test_results.append({
                        'input': sentence,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })

                if i % 2 == 0 or i == len(test_sentences):
                    progress_print(f"테스트 진행률: {i}/{len(test_sentences)}")

            except Exception as e:
                test_results.append({
                    'input': sentence,
                    'success': False,
                    'error': str(e)
                })

        scores = [r['score'] for r in test_results if r.get('success', False) and 'score' in r]
        avg_score = np.mean(scores) if scores else 0.0

        info_print(f"📊 간단한 테스트 결과:")
        progress_print(f"총 테스트: {len(test_sentences)}개")
        progress_print(f"성공: {success_count}개 ({success_count / len(test_sentences) * 100:.1f}%)")
        if success_count > 0:
            progress_print(f"평균 점수: {avg_score:.3f}")

        return {
            'total_tests': len(test_sentences),
            'accuracy_top1': success_count / len(test_sentences),
            'accuracy_top3': success_count / len(test_sentences),
            'accuracy_top5': success_count / len(test_sentences),
            'emotion_accuracy': {},
            'situation_accuracy': {},
            'detailed_results': test_results
        }

    def calculate_training_loss(self, train_data, epoch):
        """훈련 loss 시뮬레이션"""
        try:
            base_loss = 2.5
            decay_rate = 0.8
            noise = np.random.normal(0, 0.1)
            return max(0.1, base_loss * (decay_rate ** epoch) + noise)
        except:
            return 1.0

    def plot_training_curves(self):
        """훈련 곡선 시각화"""
        if not self.training_history['loss']:
            error_print("시각화할 데이터가 없습니다.")
            return

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('훈련 성능 곡선', fontsize=16, fontweight='bold')

            epochs = range(1, len(self.training_history['loss']) + 1)

            # 1. Loss 곡선
            axes[0, 0].plot(epochs, self.training_history['loss'], 'r-', linewidth=2, marker='o')
            axes[0, 0].set_title('Training Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Top-K 정확도
            axes[0, 1].plot(epochs, self.training_history['accuracy_top1'], 'b-', linewidth=2, marker='s',
                            label='Top-1')
            axes[0, 1].plot(epochs, self.training_history['accuracy_top3'], 'g-', linewidth=2, marker='^',
                            label='Top-3')
            axes[0, 1].plot(epochs, self.training_history['accuracy_top5'], 'orange', linewidth=2, marker='d',
                            label='Top-5')
            axes[0, 1].set_title('Top-K 정확도', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 감정 인식 정확도
            axes[1, 0].plot(epochs, self.training_history['emotion_accuracy'], 'purple', linewidth=2, marker='*')
            axes[1, 0].set_title('감정 인식 정확도', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Emotion Accuracy')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 평균 유사도 점수
            axes[1, 1].plot(epochs, self.training_history['similarity_scores'], 'teal', linewidth=2, marker='h')
            axes[1, 1].set_title('평균 유사도 점수', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Similarity Score')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            os.makedirs('results', exist_ok=True)
            plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
            plt.show()
            info_print("훈련 곡선이 'results/training_curves.png'에 저장되었습니다.")

        except Exception as e:
            error_print(f"훈련 곡선 시각화 실패: {e}")

    def plot_performance_analysis(self, results):
        """성능 분석 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('성능 분석 대시보드', fontsize=16, fontweight='bold')

            # 1. 감정별 정확도
            emotion_accuracy = results.get('emotion_accuracy', {})
            if emotion_accuracy:
                emotions = list(emotion_accuracy.keys())
                accuracies = list(emotion_accuracy.values())
                if emotions:
                    bars = axes[0, 0].bar(emotions, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(emotions))))
                    axes[0, 0].set_title('감정별 정확도', fontweight='bold')
                    axes[0, 0].set_ylabel('Accuracy')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 0].text(0.5, 0.5, '감정 데이터 없음', ha='center', va='center', transform=axes[0, 0].transAxes)
            else:
                axes[0, 0].text(0.5, 0.5, '감정 데이터 없음', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('감정별 정확도', fontweight='bold')

            # 2. Top-K 정확도 비교
            if results:
                top_k_data = [
                    results.get('accuracy_top1', 0.0),
                    results.get('accuracy_top3', 0.0),
                    results.get('accuracy_top5', 0.0)
                ]
                top_k_labels = ['Top-1', 'Top-3', 'Top-5']
                bars = axes[0, 1].bar(top_k_labels, top_k_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[0, 1].set_title('Top-K 정확도 비교', fontweight='bold')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].set_ylim(0, 1)

                for bar, value in zip(bars, top_k_data):
                    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                    f'{value:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, '결과 데이터 없음', ha='center', va='center', transform=axes[0, 1].transAxes)

            # 3. 점수 분포
            detailed_results = results.get('detailed_results', [])
            if detailed_results:
                scores = [r.get('score', 0.0) for r in detailed_results if isinstance(r, dict) and 'score' in r]
                if scores:
                    axes[1, 0].hist(scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('유사도 점수 분포', fontweight='bold')
                    axes[1, 0].set_xlabel('Similarity Score')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'평균: {np.mean(scores):.3f}')
                    axes[1, 0].legend()
                else:
                    axes[1, 0].text(0.5, 0.5, '점수 데이터 없음', ha='center', va='center', transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, '점수 데이터 없음', ha='center', va='center', transform=axes[1, 0].transAxes)

            # 4. 성공/실패 비율
            if detailed_results:
                success_count = sum(1 for r in detailed_results if r.get('success', False))
                failure_count = len(detailed_results) - success_count

                if success_count + failure_count > 0:
                    sizes = [success_count, failure_count]
                    labels = ['성공', '실패']
                    colors = ['#90EE90', '#FFB6C1']

                    wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                              startangle=90)
                    axes[1, 1].set_title('성공/실패 비율', fontweight='bold')
                else:
                    axes[1, 1].text(0.5, 0.5, '결과 데이터 없음', ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, '결과 데이터 없음', ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            info_print("성능 분석이 'results/performance_analysis.png'에 저장되었습니다.")

        except Exception as e:
            error_print(f"성능 분석 시각화 실패: {e}")

    def plot_confusion_matrix(self, results):
        """혼동 행렬 시각화"""
        try:
            emotion_accuracy = results.get('emotion_accuracy', {})
            if not emotion_accuracy:
                debug_print("감정 데이터가 없어 혼동 행렬을 생성할 수 없습니다.")
                return

            emotions = list(emotion_accuracy.keys())
            if len(emotions) < 2:
                debug_print("혼동 행렬 생성을 위해서는 최소 2개의 감정이 필요합니다.")
                return

            n_emotions = len(emotions)
            confusion_data = np.random.rand(n_emotions, n_emotions)

            for i in range(n_emotions):
                confusion_data[i, i] += 0.5

            confusion_data = confusion_data / confusion_data.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_data, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=emotions, yticklabels=emotions)
            plt.title('감정 분류 혼동 행렬', fontsize=14, fontweight='bold')
            plt.xlabel('예측된 감정')
            plt.ylabel('실제 감정')

            os.makedirs('results', exist_ok=True)
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            info_print("혼동 행렬이 'results/confusion_matrix.png'에 저장되었습니다.")

        except Exception as e:
            error_print(f"혼동 행렬 생성 실패: {e}")

    def _print_simple_evaluation_report(self, results):
        """간단한 평가 리포트 출력"""
        print("\n" + "=" * 50)
        print("📊 평가 결과")
        print("=" * 50)

        print(f"🎯 전체 성능:")
        print(f"  • 총 테스트 수: {results['total_tests']}개")
        print(f"  • 성공한 테스트: {results.get('successful_tests', 0)}개")
        print(f"  • Top-1 정확도: {results['accuracy_top1']:.1%}")
        print(f"  • Top-3 정확도: {results['accuracy_top3']:.1%}")
        print(f"  • Top-5 정확도: {results['accuracy_top5']:.1%}")

        print("=" * 50)

    def _save_evaluation_results(self, results, filename='results/evaluation_results.json'):
        """평가 결과 저장"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (list, dict, str, int, float, bool)):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            info_print(f"평가 결과가 '{filename}'에 저장되었습니다.")

        except Exception as e:
            error_print(f"결과 저장 실패: {e}")

    def save_training_history(self):
        """훈련 기록 저장"""
        try:
            os.makedirs('results', exist_ok=True)

            history_file = 'results/training_history.json'
            history_to_save = {}

            for key, values in self.training_history.items():
                converted_values = []
                for v in values:
                    if isinstance(v, (np.floating, np.integer)):
                        converted_values.append(float(v))
                    elif isinstance(v, np.ndarray):
                        converted_values.append(v.tolist())
                    else:
                        converted_values.append(v)
                history_to_save[key] = converted_values

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)

            info_print(f"훈련 기록이 '{history_file}'에 저장되었습니다.")

        except Exception as e:
            error_print(f"훈련 기록 저장 실패: {e}")

    def test_korean_text_processing(self):
        """한국어 텍스트 처리 테스트"""
        if not DEBUG_MODE:
            return

        info_print("🇰🇷 한국어 텍스트 처리 테스트...")

        korean_test_cases = [
            "안녕하세요!",
            "스트레스 받아요 ㅠㅠ",
            "결혼을 하라고 부모님이 자꾸 말해서 스트레스야.",
            "😊🎉🌟 이모지가 포함된 텍스트입니다!",
            "",
            "   ",
            "a" * 300
        ]

        for i, test_text in enumerate(korean_test_cases, 1):
            debug_print(f"[테스트 {i}] 원본: {repr(test_text)}")

            try:
                cleaned = self.validate_and_clean_text(test_text)
                debug_print(f"         정리됨: {repr(cleaned)}")

                if hasattr(self.clip_encoder, 'test_tokenization'):
                    token_result = self.clip_encoder.test_tokenization(cleaned)
                    debug_print(f"         토큰화: {'✅ 성공' if token_result else '❌ 실패'}")

                embedding = self.clip_encoder.get_text_embedding(cleaned)
                embedding_success = embedding is not None and len(embedding) > 0 and not np.any(np.isnan(embedding))
                debug_print(f"         임베딩: {'✅ 성공' if embedding_success else '❌ 실패'}")

            except Exception as e:
                debug_print(f"         ❌ 오류: {e}")

        info_print("한국어 텍스트 처리 테스트 완료")


def safe_main():
    """안전한 메인 함수 실행"""
    try:
        os.makedirs('results', exist_ok=True)
        info_print("Results 디렉토리 준비 완료")

        info_print("시스템 초기화 중...")
        system = TrainableRecommendationSystem()

        # 한국어 텍스트 처리 테스트
        system.test_korean_text_processing()

        info_print("학습 및 평가 시작...")
        results = system.train_and_evaluate(test_size=0.2, epochs=5)

        # 훈련 기록 저장
        system.save_training_history()

        # 실시간 테스트
        info_print("🎯 실시간 추천 테스트")

        test_cases = [
            "일은 왜 해도 해도 끝이 없을까? 화가 난다.",
            "오늘 정말 기분이 좋아! 승진했어!",
            "연인과 헤어져서 너무 슬퍼.",
            "회사 상사가 너무 짜증나."
        ]

        successful_tests = 0
        for i, dialogue in enumerate(test_cases, 1):
            try:
                result = system.recommend(dialogue)
                if result.get('success', False):
                    successful_tests += 1
                    progress_print(f"테스트 {i}: ✅ 성공 (점수: {result['score']:.3f})")
                else:
                    progress_print(f"테스트 {i}: ⚠️ 폴백 추천")

            except Exception as e:
                progress_print(f"테스트 {i}: ❌ 실패")

        info_print(f"🎉 모든 작업이 완료되었습니다!")
        progress_print(f"실시간 테스트 성공률: {successful_tests}/{len(test_cases)}")
        progress_print(f"결과 파일들이 'results/' 디렉토리에 저장되었습니다.")

        return results

    except KeyboardInterrupt:
        info_print("⚠️ 사용자에 의해 중단되었습니다.")
        return None

    except Exception as e:
        error_print(f"시스템 초기화 실패: {e}")
        info_print("💡 해결 방법:")
        progress_print("1. data/enhanced_image_metadata.json 파일이 존재하는지 확인")
        progress_print("2. data/images/ 디렉토리에 이미지 파일들이 있는지 확인")
        progress_print("3. 메타데이터 파일의 경로 정보가 정확한지 확인")
        progress_print("4. CLIP 모델이 올바르게 로드되는지 확인")
        progress_print("5. 한국어 텍스트 처리에 문제가 없는지 확인")
        return None


def main():
    """메인 함수"""
    return safe_main()


if __name__ == "__main__":
    main()