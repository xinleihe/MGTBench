import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics


@timeit
def run_supervised_experiment(data, model, cache_dir, batch_size, DEVICE, pos_bit=0):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']
    train_preds = get_supervised_model_prediction(
        detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
    test_preds = get_supervised_model_prediction(
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
