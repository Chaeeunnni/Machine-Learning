from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_best_match(text_emb, image_embs):
    similarities = cosine_similarity([text_emb], image_embs)
    best_idx = np.argmax(similarities)
    return best_idx, similarities[0][best_idx]