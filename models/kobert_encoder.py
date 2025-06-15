from transformers import BertModel, BertTokenizer
# from transformers import AutoModel, AutoTokenizer
import torch

class KoBERTEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")  # 변경
        self.model = BertModel.from_pretrained("monologg/kobert").to(device)  # 변경

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()[0]
