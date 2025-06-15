import os

def rename_images_globally(base_folder, prefix="img"):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    counter = 1
    temp_suffix = "_TEMP_RENAME"

    # 1차 리네이밍: 임시 이름 부여 (중복 회피용)
    for root, dirs, files in os.walk(base_folder):
        image_files = sorted([f for f in files if f.lower().endswith(image_extensions)])
        for filename in image_files:
            ext = os.path.splitext(filename)[1]
            temp_name = f"{prefix}{counter:03d}{temp_suffix}{ext}"
            src = os.path.join(root, filename)
            dst = os.path.join(root, temp_name)

            os.rename(src, dst)
            print(f"[1차] {filename} → {temp_name}")
            counter += 1

    # 2차 리네이밍: 최종 이름 부여
    counter = 1
    for root, dirs, files in os.walk(base_folder):
        image_files = sorted([f for f in files if temp_suffix in f])
        for filename in image_files:
            ext = os.path.splitext(filename)[1]
            final_name = f"{prefix}{counter:03d}{ext}"
            src = os.path.join(root, filename)
            dst = os.path.join(root, final_name)

            os.rename(src, dst)
            print(f"[2차] {filename} → {final_name}")
            counter += 1

# ✅ 사용 예시
rename_images_globally("D:/2025-1(4-2)/MLproject/data/images")
