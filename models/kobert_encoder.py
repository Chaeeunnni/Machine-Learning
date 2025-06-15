from transformers import BertModel
# , BertTokenizer)
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

class KoBERTEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = tokenizer.from_pretrained("monologg/kobert")
        self.model = BertModel.from_pretrained("monologg/kobert").to(device)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()[0]
