import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--best', type=float, help='the current best f1 for the data and detectLLM', required=True, default=1)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    datatype = args.dataset
    llmname = args.detectLLM
    seed = args.seed
    size = args.data_size
    cut_length = args.cut_length
    save = args.save
    task = args.task
    folder = args.folder
    best = args.best
    eval = args.eval

    if eval:
        setup_seed(seed)
        metric1 = AutoDetector.from_detector_name('LM-D', 
                                                model_name_or_path=distilbert)
        
        metric1.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{datatype}_{llmname}_{task}').to('cuda')
        experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric1])

        data = load(datatype, llmname, cut_length=cut_length, task=task)
        data['train']['text'] = data['train']['text'][:]
        data['train']['label'] = data['train']['label'][:]
        # data['test']['text'] = data['test']['text'][:size]
        # data['test']['label'] = data['test']['label'][:size]

        experiment.load_data(data)
        res = experiment.launch(need_finetune=False)
        print(res[0].train)
        print(res[0].test)
        exit()
        

    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = f'./{folder}/{datatype}_{llmname}_{seed}_{cut_length}_{size}.txt'

    print(f"------ Running {datatype} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {size} ------")
    with open(output_path, "a") as file:
        print(f"------ Running {datatype} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {size} ------", file=file)


    if args.model == 'bert':
        model_name = bert
    elif args.model == 'roberta':
        model_name = roberta
    else:
        model_name = distilbert

    setup_seed(seed)

    torch.cuda.empty_cache()

    metric1 = AutoDetector.from_detector_name('LM-D', 
                                                model_name_or_path=model_name)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric1])

    data = load(datatype, llmname, cut_length=cut_length, task=task)
    data['train']['text'] = data['train']['text'][:size]
    data['train']['label'] = data['train']['label'][:size]
    # TODO: remove size!
    # data['test']['text'] = data['test']['text'][:size]
    # data['test']['label'] = data['test']['label'][:size]

    experiment.load_data(data)
    res = experiment.launch(**config)
    print(res[0].train)
    print(res[0].test)
    with open(output_path, "a") as file:
        print(res[0].train, file=file)
        print(res[0].test, file=file)

    cur_f1 = res[0].test.f1
    if save:
        if cur_f1 > best:
            print('Saving model')
            save_path = '/data1/zzy/finetuned'
            name = f'{datatype}_{llmname}_{task}_{args.model}'
            metric1.model.save_pretrained(f'{save_path}/{name}')
        # reload the model
        # metric_temp = AutoDetector.from_detector_name('LM-D', 
        #                                     model_name_or_path=model_name)
        # metric_temp.model = AutoModelForSequenceClassification.from_pretrained(f'{save_path}/{name}').to('cuda')
        # experiment_temp = AutoExperiment.from_experiment_name('supervised',detector=[metric_temp])
        # experiment_temp.load_data(data)
        # res = experiment_temp.launch(need_finetune=False)
        # print(res[0].test)