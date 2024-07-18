import os
import requests
from tqdm import tqdm
from ..utils import timeit, cal_metrics
from ..auto import BaseDetector
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

class GPTZeroDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.api_key = kargs.get('api_key', None)
        if not self.api_key:
            raise ValueError('You should pass an api_key before using the GPTZero Detector')
        self.api = GPTZeroAPI(self.api_key)

    def detect(self, text, **kargs):
        pred_prob = [gptzero_api.text_predict(x)['documents'][0]["completely_generated_prob"] for x in tqdm(text)]
        return [round(_) for _ in pred_prob]




if __name__ == '__main__':
    api_key = ''  # Your API Key from https://gptzero.me
    gptzero_api = GPTZeroAPI(api_key)

    document = 'Hello world!'
    response = gptzero_api.text_predict(document)
    print(response)
    # {'documents': [{'average_generated_prob': 0, 'completely_generated_prob': 0.11111111111111108, 'overall_burstiness': 0, 'paragraphs': [{'completely_generated_prob': 0.11111111111111108, 'num_sentences': 1, 'start_sentence_index': 0}], 'sentences': [{'generated_prob': 0, 'perplexity': 270, 'sentence': 'Hello world!'}]}]}
