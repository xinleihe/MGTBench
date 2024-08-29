import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch
import argparse

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

config = {'need_finetune': True,
          'need_save': False,
          'epochs': 1
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Art")
    parser.add_argument('--detectLLM', type=str, default="Moonshot")
    parser.add_argument('--method', type=str, default="LM-D")
    parser.add_argument('--cut_length', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_size', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    setup_seed(seed)

    torch.cuda.empty_cache()
    model_name_or_path = '/data1/models/distilbert-base-uncased'
    metric1 = AutoDetector.from_detector_name('LM-D', 
                                                model_name_or_path=model_name_or_path)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric1])

    datatype = args.dataset
    llmname = args.detectLLM
    size = args.data_size
    cut_length = args.cut_length

    data = load(datatype, llmname, cut_length=cut_length)
    data['train']['text'] = data['train']['text'][:size]
    data['train']['label'] = data['train']['label'][:size]
    data['test']['text'] = data['test']['text'][:size]
    data['test']['label'] = data['test']['label'][:size]

    experiment.load_data(data)
    res = experiment.launch(**config)
    print(res[0].train)
    print(res[0].test)
    del experiment
    del metric1
    torch.cuda.empty_cache()