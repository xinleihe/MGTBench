import os
import torch
import argparse
import json

from transformers import AutoModelForSequenceClassification
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

config = {'need_finetune': True,
          'need_save': False,
          'epochs': 1
        }

distilbert = '/data1/models/distilbert-base-uncased'
roberta = '/data1/zzy/roberta-base'
bert = '/data1/zzy/bert-base-uncased'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Art")
    parser.add_argument('--task', type=str, choices=['task2','task2_gen', 'task3'])
    parser.add_argument('--detectLLM', type=str, default="Moonshot")
    parser.add_argument('--model', type=str, default="distilbert")
    parser.add_argument('--cut_length', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_size', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--best', type=float, help='the current best f1 for the data and detectLLM', default=1)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--match_data', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    datatype = args.dataset
    task = args.task
    llmname = args.detectLLM
    cut_length = args.cut_length
    size = args.data_size
    seed = args.seed
    save = args.save
    best = args.best
    folder = args.folder
    match_data = args.match_data
    eval = args.eval

    if args.model == 'bert':
        model_name = bert
    elif args.model == 'roberta':
        model_name = roberta
    else:
        model_name = distilbert

    match_tag = '_match' if match_data else ''
    if eval:
        setup_seed(seed)
        metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_name)
        metric1.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{datatype}_{llmname}_{task}{match_tag}_{args.model}').to('cuda')
        experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
        data = load(datatype, llmname, cut_length=cut_length, task=task, match=match_data)
        experiment.load_data(data)
        res = experiment.launch(need_finetune=False)
        print('----------')
        print('Category:', datatype)
        print('DetectLLM:', llmname)
        print('Task:', task)
        print('Match data:', match_data)
        print('Model:', args.model)
        print(res[0].train)
        print(res[0].test)
        print('----------')
        exit()
        
    if not os.path.exists(folder):
        raise ValueError(f'{folder} does not exist')

    output_path = f'./{folder}/{datatype}_{llmname}.txt' # one file for each subject

    print(f"------ Running {datatype} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {size} ------")
    # with open(output_path, "a") as file:
    #     print(f"------ Running {datatype} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {size} ------", file=file)

    setup_seed(seed)

    torch.cuda.empty_cache()

    metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_name)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric1])

    data = load(datatype, llmname, cut_length=cut_length, task=task, match=match_data)
    data['train']['text'] = data['train']['text'][:size]
    data['train']['label'] = data['train']['label'][:size]
    # given size may be larger than the maximum length of the dataset, record the maximum length
    max_len = len(data['train']['text'])
    if size > max_len:
        print(f"Warning: data_size {size} is larger than the maximum length {max_len} of the dataset. Using the maximum length instead.")
        size = max_len

    experiment.load_data(data)
    res = experiment.launch(**config)
    print(res[0].train)
    print(res[0].test)
    # with open(output_path, "a") as file:
    #     print(res[0].train, file=file)
    #     print(res[0].test, file=file)
    #     print('\n', file=file)

    # serialize the train and test Metric object
    temp = {'cut_length': cut_length,
            'data_size': size,
            'seed': seed,
            'train': res[0].train.__dict__,
            'test': res[0].test.__dict__
            }
    with open(f'{folder}/experiment.json', 'w') as f:
        json.dump(temp, f)

    cur_f1 = res[0].test.f1
    if save:
        if cur_f1 > best:
            print('Saving model')
            save_path = '/data1/zzy/finetuned'
            name = f'{datatype}_{llmname}_{task}{match_tag}_{args.model}'
            metric1.model.save_pretrained(f'{save_path}/{name}')