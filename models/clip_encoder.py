from PIL import Image
import torch
import clip
import os

class CLIPEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.cpu().numpy()[0]

    def get_text_embedding(self, text):
        """텍스트를 임베딩으로 변환 - 추가된 메서드"""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.cpu().numpy()[0]

    def encode_text(self, text):
        """get_text_embedding의 별칭"""
        return self.get_text_embedding(text)

    def get_image_embedding(self, image_path):
        """encode_image의 별칭"""
        return self.encode_image(image_path)

    def encode_all_images(self, metadata, image_root):
        embeddings = []
        filenames = []
        for item in metadata:
            image_path = os.path.join(image_root, item['filepath'])
            try:
                emb = self.encode_image(image_path)
                embeddings.append(emb)
                filenames.append(item['filename'])
            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")
        return embeddings, filenames
