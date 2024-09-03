import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import json
import argparse
import torch

from transformers import AutoModelForSequenceClassification
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot']

distilbert = '/data1/models/distilbert-base-uncased'
roberta = '/data1/zzy/roberta-base'
bert = '/data1/zzy/bert-base-uncased'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=category, default="Art")
    parser.add_argument('--detectLLM', type=str, choices=llms, default="Moonshot")
    parser.add_argument('--task', type=str, choices=['old', 'task2','task2_gen', 'task3'],)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    detectLLM = args.detectLLM
    task = args.task
    eval_all = args.all

    with open(f'{task}_final/best_hyperparams.json', 'r') as f:
        best_hyperparams = json.load(f)

    if eval_all:
        for cat in category:
            for llm in llms:
                best_model = best_hyperparams[cat][llm][0]
                best_cut_length = best_hyperparams[cat][llm][2]

                if best_model == 'bert':
                    model_name = bert
                elif best_model == 'roberta':
                    model_name = roberta
                else:
                    model_name = distilbert
                metric = AutoDetector.from_detector_name('LM-D', 
                                                model_name_or_path=model_name)
        
                metric.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{dataset}_{detectLLM}_{task}_{best_model}').to('cuda')
                experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric])

                data = load(cat, llm, cut_length=best_cut_length, task=task)
                experiment.load_data(data)
                res = experiment.launch(need_finetune=False)
                print('----------')
                print('Category:', cat)
                print('DetectLLM:', llm)
                print('Task:', task)
                print('Model:', best_model)
                print(res[0].train)
                print(res[0].test)
                print('----------')
                del metric
                del experiment
                torch.cuda.empty_cache()

    else:
        best_model = best_hyperparams[dataset][detectLLM][0]
        best_cut_length = best_hyperparams[dataset][detectLLM][2]

        # setup_seed(420)
        if best_model == 'bert':
            model_name = bert
        elif best_model == 'roberta':
            model_name = roberta
        else:
            model_name = distilbert
        metric = AutoDetector.from_detector_name('LM-D', 
                                                model_name_or_path=model_name)
        
        metric.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{dataset}_{detectLLM}_{task}_{best_model}').to('cuda')
        experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric])

        data = load(dataset, detectLLM, cut_length=best_cut_length, task=task)
        # data['test']['text'] = data['test']['text'][:1000]
        # data['test']['label'] = data['test']['label'][:1000]
        experiment.load_data(data)
        res = experiment.launch(need_finetune=False)
        print('----------')
        print('Category:', dataset)
        print('DetectLLM:', detectLLM)
        print('Task:', task)
        print('Model:', best_model)
        print(res[0].train)
        print(res[0].test)
        print('----------')

    


                


        
