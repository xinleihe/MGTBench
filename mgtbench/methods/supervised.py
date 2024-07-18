import numpy as np
import transformers
import torch
from tqdm import tqdm
from ..utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW
from ..auto import BaseDetector
from ..loading import load_pretrained_supervise
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SupervisedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained_supervise(model_name_or_path, quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        if ("state_dict_path" in kargs) and ("state_dict_key" in kargs):
            self.model.load_state_dict(
                torch.load(kargs["state_dict_path"],map_location='cpu')[kargs["state_dict_key"]])
        
    def detect(self, text, **kargs):
        result = []
        pos_bit=0
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=512,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
        # print(result)
        return result if isinstance(text, list) else result[0]
    
    def finetune(self, data, config):
        batch_size = config.batch_size
        num_epochs = config.epochs
        save_path = config.save_path
        if config.pos_bit == 0:
            train_label = [1 if _ == 0 else 0 for _ in data['label']]

        train_encodings = self.tokenizer(data['text'], truncation=True, padding=True)
        train_dataset = CustomDataset(train_encodings, data['label'])

        self.model.train()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        self.model.eval()
        if config.need_save:
            self.model.save_pretrained(f'{save_path}/{self.name}')

