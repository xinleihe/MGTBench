import os
import json
import argparse
import torch
import time
import subprocess

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
    parser.add_argument('--task', type=str, choices=['old', 'task2','task2_gen', 'task3'])
    parser.add_argument('--match_data', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    detectLLM = args.detectLLM
    task = args.task
    match_data = args.match_data
    eval_all = args.all

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    match_tag = '_match' if match_data else ''
    with open(f'{task}_best/best_hyperparams{match_tag}.json', 'r') as f:
        best_hyperparams = json.load(f)

    if eval_all:
        for cat in category:
            for llm in llms:
                best_model = best_hyperparams[cat][llm]['model']
                seed = best_hyperparams[cat][llm]['seed']
                best_cut_length = best_hyperparams[cat][llm]['cut_length']

                command = f"python run_lm.py " \
                        f"--dataset {cat} " \
                        f"--detectLLM {llm} " \
                        f"--model {best_model} " \
                        f"--cut_length {best_cut_length} " \
                        f"--seed {seed} " \
                        f"--task {task} " \
                        f"--folder test " \
                        f"--match_data {match_data} " \
                        f"--eval"
                subprocess.run(command, shell=True)
                time.sleep(1)
                torch.cuda.empty_cache()

    else:
        best_model = best_hyperparams[dataset][detectLLM]['model']
        seed = best_hyperparams[dataset][detectLLM]['seed']
        best_cut_length = best_hyperparams[dataset][detectLLM]['cut_length']

        command = f"python run_lm.py " \
                f"--dataset {dataset} " \
                f"--detectLLM {detectLLM} " \
                f"--model {best_model} " \
                f"--cut_length {best_cut_length} " \
                f"--seed {seed} " \
                f"--task {task} " \
                f"--folder test " \
                f"--match_data {match_data} " \
                f"--eval"
        subprocess.run(command, shell=True)
