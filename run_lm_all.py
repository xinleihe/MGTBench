import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import argparse

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

config = {'need_finetune': True,
          'need_save': False,
          'epochs': 1
        }

distilbert = '/data1/models/distilbert-base-uncased'

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

old_category = ['Essay', 'WP', 'Reuters']
old_llms = ['ChatGLM', 'Dolly', 'ChatGPT-turbo', 'GPT4All', 'StableLM', 'Claude']

seeds = [42, 3407, 114514]
# for new dataset (task 2 or 3)
cut_sizes = [3000, 6000]
data_sizes = [3000, 6000]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, required=True, choices=['LM-D', 'll'])
    parser.add_argument('--experiment', type=str, required=True, choices=['supervised', 'threshold'])
    parser.add_argument('--data', type=str, default="old")
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    which_data = args.data
    folder = args.folder
    detector_name = args.detector
    experiment_name = args.experiment

    if not os.path.exists(folder):
        os.makedirs(folder)

    if which_data == 'old':
        for seed in seeds:
            setup_seed(seed)
            torch.cuda.empty_cache()
            for cat in old_category:
                for llmname in old_llms:
                    output_path = f'./{folder}/{cat}_{llmname}_{seed}.txt'

                    print(f"------ Running {cat} with seed {seed}, llmname {llmname} ------")
                    with open(output_path, "a") as file:
                        print(f"------ Running {cat} with seed {seed}, llmname {llmname} ------", file=file)

                    metric = AutoDetector.from_detector_name(detector_name, model_name_or_path=distilbert)
                    experiment = AutoExperiment.from_experiment_name(experiment_name, detector=[metric])

                    data = load(cat, llmname, disable=True)
                    experiment.load_data(data)
                    res = experiment.launch(**config)

                    print(res[0].train)
                    print(res[0].test)
                    with open(output_path, "a") as file:
                        print(res[0].train, file=file)
                        print(res[0].test, file=file)

                    del experiment
                    del metric
                    torch.cuda.empty_cache()

    elif which_data == 'task2' or which_data == 'task3':
        for seed in seeds:
            setup_seed(seed)
            torch.cuda.empty_cache()
            for cat in category:
                for cut_length in cut_sizes:
                    for num_data in data_sizes:
                        llmname = 'Moonshot'
                        output_path = f'./{folder}/{cat}_{llmname}_{seed}_{cut_length}_{num_data}.txt'

                        print(f"------ Running {cat} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {num_data} ------")
                        with open(output_path, "a") as file:
                            print(f"------ Running {cat} and model {llmname} with seed {seed}, cut_length {cut_length}, data_size {num_data} ------", file=file)

                        metric = AutoDetector.from_detector_name('LM-D', model_name_or_path=distilbert)
                        experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric])

                        data = load(cat, llmname, cut_length=cut_length, disable=True, task=which_data)
                        data['train']['text'] = data['train']['text'][:num_data]
                        data['train']['label'] = data['train']['label'][:num_data]
                        data['test']['text'] = data['test']['text'][:num_data]
                        data['test']['label'] = data['test']['label'][:num_data]

                        experiment.load_data(data)
                        res = experiment.launch(**config)
                        print(res[0].train)
                        print(res[0].test)
                        with open(output_path, "a") as file:
                            print(res[0].train, file=file)
                            print(res[0].test, file=file)

                        del experiment
                        del metric
                        torch.cuda.empty_cache()