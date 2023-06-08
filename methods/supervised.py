import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW


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


@timeit
def run_supervised_experiment(data, model, cache_dir, batch_size, DEVICE, pos_bit=0, finetune=False, num_labels=2, epochs=3):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=num_labels, cache_dir=cache_dir, ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs=epochs)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    # detector.save_pretrained(".cache/lm-d-xxx", from_pt=True)

    if num_labels == 2:
        train_preds = get_supervised_model_prediction(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        'train': train_preds,
        'test': test_preds,
    }
    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res
    print(f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(f"{model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


def get_supervised_model_prediction(model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating real"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(batch_data, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_multi_classes(model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating real"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(batch_data, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


def fine_tune_model(model, tokenizer, data, batch_size, DEVICE, pos_bit=1, num_labels=2, epochs=3):

    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    model.train()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
    model.eval()
