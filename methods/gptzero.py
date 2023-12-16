import os
import requests
from tqdm import tqdm
from methods.utils import timeit, cal_metrics

# from https://github.com/Haste171/gptzero


class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'

    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()


def run_gptzero_experiment(data, api_key):

    gptzero_api = GPTZeroAPI(api_key)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    train_pred_prob = [gptzero_api.text_predict(
        _)['documents'][0]["completely_generated_prob"] for _ in tqdm(train_text)]
    test_pred_prob = [gptzero_api.text_predict(
        _)['documents'][0]["completely_generated_prob"] for _ in tqdm(test_text)]
    train_pred = [round(_) for _ in train_pred_prob]
    test_pred = [round(_) for _ in test_pred_prob]

    acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(
        train_label, train_pred, train_pred_prob)
    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
        test_label, test_pred, test_pred_prob)

    print(
        f"GPTZero acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"GPTZero acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return {
        'name': 'GPTZero',
        'predictions': {'train': train_pred_prob, 'test': test_pred_prob},
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


if __name__ == '__main__':
    api_key = ''  # Your API Key from https://gptzero.me
    gptzero_api = GPTZeroAPI(api_key)

    document = 'Hello world!'
    response = gptzero_api.text_predict(document)
    print(response)
    # {'documents': [{'average_generated_prob': 0, 'completely_generated_prob': 0.11111111111111108, 'overall_burstiness': 0, 'paragraphs': [{'completely_generated_prob': 0.11111111111111108, 'num_sentences': 1, 'start_sentence_index': 0}], 'sentences': [{'generated_prob': 0, 'perplexity': 270, 'sentence': 'Hello world!'}]}]}
