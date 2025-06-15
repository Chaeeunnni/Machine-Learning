from openai import OpenAI
import base64
import json
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# OpenAI API 키 설정
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

def describe_image_with_gpt4v(image_path):
    """GPT-4V를 사용해서 이미지 설명 생성"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 짤 이미지를 한국어로 상세히 설명해주세요. 다음 형식으로 답변해주세요:\n1. 감정 표현: \n2. 상황/맥락: \n3. 캐릭터 행동: \n4. 사용하기 좋은 대화 상황: "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def extract_usage_context_from_description(description):
    """GPT-4V 설명에서 사용 맥락 추출"""
    if not description:
        return []

    # 간단한 키워드 매칭으로 사용 맥락 추출
    context_keywords = {
        "축하": ["축하", "성공", "기쁨", "달성"],
        "위로": ["슬픔", "위로", "힘듦", "우울"],
        "격려": ["응원", "격려", "화이팅", "파이팅"],
        "공감": ["이해", "공감", "동감"],
        "조언": ["조언", "충고", "도움"],
        "유머": ["웃김", "재미", "농담", "개그"],
        "분노": ["화남", "짜증", "분노"],
        "놀람": ["놀람", "당황", "충격"],
        "업무": ["일", "업무", "직장", "회사"],
        "연애": ["사랑", "연애", "데이트", "커플"],
        "친구": ["친구", "우정", "모임"],
        "가족": ["가족", "부모", "형제"]
    }

    contexts = []
    for context, keywords in context_keywords.items():
        if any(keyword in description for keyword in keywords):
            contexts.append(context)

    return contexts


def process_single_image(image_info, base_data_path):
    """단일 이미지 처리"""
    # image_metadata.json의 filepath: "기쁨/감사하는/img003.jpg"
    # 실제 파일 경로: MLproject/data/images/기쁨/감사하는/img003.jpg
    full_image_path = os.path.join(base_data_path, "images", image_info['filepath'])

    print(f"처리 중인 이미지: {image_info['filepath']}")
    print(f"전체 경로: {full_image_path}")

    # 파일이 존재하는지 확인
    if not os.path.exists(full_image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {full_image_path}")
        return {
            **image_info,
            "text_description": "파일을 찾을 수 없음",
            "usage_context": [],
            "processing_status": "file_not_found"
        }

    # GPT-4V로 설명 생성
    print("🔄 GPT-4V로 이미지 분석 중...")
    description = describe_image_with_gpt4v(full_image_path)

    if description:
        print("✅ 설명 생성 완료")
        # 설명에서 사용 맥락 추출
        usage_contexts = extract_usage_context_from_description(description)
    else:
        print("❌ 설명 생성 실패")
        usage_contexts = []

    # 기존 메타데이터에 설명 추가
    enhanced_metadata = {
        **image_info,
        "text_description": description if description else "설명 생성 실패",
        "usage_context": usage_contexts,
        "processing_status": "success" if description else "failed",
        "processed_path": full_image_path  # 디버깅용
    }

    # API 호출 제한을 위한 잠시 대기
    time.sleep(1)

    return enhanced_metadata


def verify_image_structure(base_data_path):
    """이미지 폴더 구조 확인"""
    images_path = os.path.join(base_data_path, "images")

    if not os.path.exists(images_path):
        print(f"❌ images 폴더를 찾을 수 없습니다: {images_path}")
        return False

    print(f"✅ images 폴더 확인: {images_path}")

    # 대분류 폴더들 확인
    categories = []
    for item in os.listdir(images_path):
        item_path = os.path.join(images_path, item)
        if os.path.isdir(item_path):
            categories.append(item)
            print(f"  📁 대분류: {item}")

            # 소분류 폴더들 확인
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    # 이미지 파일 개수 확인
                    image_files = [f for f in os.listdir(subitem_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                    print(f"    📂 소분류: {subitem} ({len(image_files)}개 이미지)")

    return True


def batch_process_images():
    """배치로 모든 이미지 처리"""
    # 프로젝트 루트 경로 설정
    project_root = "/Users/nahyeon/2025-1/기계학습개론/Machine-Learning"  # 실제 프로젝트 폴더명에 맞게 수정

    # 메타데이터 파일 경로
    metadata_path = os.path.join(project_root, "data", "image_metadata.json")
    output_path = os.path.join(project_root, "data", "enhanced_image_metadata.json")

    # 데이터 폴더 경로
    data_path = os.path.join(project_root, "data")

    print("=" * 50)
    print("🚀 이미지 메타데이터 자동 생성 시작")
    print("=" * 50)
    print(f"프로젝트 루트: {project_root}")
    print(f"메타데이터 파일: {metadata_path}")
    print(f"출력 파일: {output_path}")
    print()

    # 폴더 구조 확인
    print("📁 이미지 폴더 구조 확인 중...")
    if not verify_image_structure(data_path):
        return
    print()

    # 메타데이터 파일 존재 확인
    if not os.path.exists(metadata_path):
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        return

    # 메타데이터 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        image_data = json.load(f)

    print(f"📊 총 {len(image_data)}개의 이미지를 처리합니다.")
    print()

    # 순차 처리 (API 제한을 고려)
    enhanced_data = []
    successful_count = 0
    failed_count = 0

    for i, image_info in enumerate(tqdm(image_data, desc="Processing images")):
        print(f"\n[{i + 1}/{len(image_data)}] ", end="")
        result = process_single_image(image_info, data_path)
        enhanced_data.append(result)

        if result.get('processing_status') == 'success':
            successful_count += 1
        else:
            failed_count += 1

        # 10개마다 중간 저장
        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 중간 저장 완료: {i + 1}개 처리됨 (성공: {successful_count}, 실패: {failed_count})")

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("🎉 처리 완료!")
    print(f"✅ 성공: {successful_count}개")
    print(f"❌ 실패: {failed_count}개")
    print(f"💾 결과 파일: {output_path}")
    print("=" * 50)


# 실행
if __name__ == "__main__":
    batch_process_images()