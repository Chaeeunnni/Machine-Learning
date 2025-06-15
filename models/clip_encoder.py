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
