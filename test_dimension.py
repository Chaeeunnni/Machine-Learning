from models.hybrid_encoder import HybridEncoder

encoder = HybridEncoder()

test_text = "테스트 문장입니다."
print("=== 차원 테스트 ===")

try:
    kobert_emb = encoder.kobert_encoder.encode_text(test_text)
    print(f"KoBERT 임베딩 차원: {kobert_emb.shape}")
except Exception as e:
    print(f"KoBERT 에러: {e}")

try:
    clip_emb = encoder.clip_encoder.get_text_embedding(test_text)
    print(f"CLIP 임베딩 차원: {clip_emb.shape}")
except Exception as e:
    print(f"CLIP 에러: {e}")

try:
    combined_emb = encoder.get_text_embedding(test_text)
    print(f"결합 임베딩 차원: {combined_emb.shape}")
except Exception as e:
    print(f"결합 에러: {e}")