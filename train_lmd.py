import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import time
import json
import re
import argparse
import subprocess

config = {'need_finetune': True,
          'need_save': False,
          'epochs': 1
        }

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot']

seeds = [420, 3407, 114514]
# seeds = [2024, 777, 12345]

# for new dataset (task 2 or 3)
cut_sizes = [500, 700, 1500]
data_sizes = [500, 800, 2000]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert', choices=['distilbert', 'roberta', 'bert'])
    parser.add_argument('--task', type=str, default="old")
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--previous_best', type=bool, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()

    which_task = args.task
    folder = args.folder
    config['epochs'] = args.epochs
    model = args.model
    save = args.save
    previous_best = args.previous_best

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    # maintain the best f1 for each category and detectLLM
    if previous_best:
        assert os.path.exists(f'{folder}/best_f1.json')
        assert os.path.exists(f'{folder}/best_hyperparams.json')
        with open(f'{folder}/best_f1.json', 'r') as f:
            best_f1 = json.load(f)
        with open(f'{folder}/best_hyperparams.json', 'r') as f:
            best_hyperparams = json.load(f)
    else:
        assert os.path.exists(f'{folder}/best_f1.json') == False
        assert os.path.exists(f'{folder}/best_hyperparams.json') == False
        best_f1 = {}
        for cat in category:
            best_f1[cat] = 0
            for llm in llms:
                best_f1[cat] = {llm: 0}

        best_hyperparams = {}
        for cat in category:
            for llm in llms:
                best_hyperparams[cat] = {llm: []}

    for cat in category:
        for llmname in llms:
            for cut_length in cut_sizes:
                for num_data in data_sizes:
                    for seed in seeds:
                        command = f"python run_lm.py --dataset {cat} --detectLLM {llmname} --model {model} --cut_length {cut_length} --data_size {num_data} --seed {seed} --save {save} --task {which_task} --best {best_f1[cat][llmname]} --folder {folder}"
                        subprocess.run(command, shell=True)
                        # check f1 to decide whether to save the model
                        time.sleep(2)

                        with open(os.path.join(folder, f'{cat}_{llmname}_{seed}_{cut_length}_{num_data}.txt'), 'r') as file:
                            lines = file.readlines()
                            line = lines[-1]
                            matches = re.findall(r"(f1)=([\d\.]+)", line)
                            try:
                                f1_score = round(float(matches[0][1]), 3)
                                if f1_score > best_f1[cat][llmname]:
                                    best_f1[cat][llmname] = f1_score
                                    best_hyperparams[cat][llmname] = [model, seed, cut_length, num_data, f1_score]

                                    with open(f'{folder}/best_f1.json', 'w') as f:
                                        json.dump(best_f1, f)
                                    with open(f'{folder}/best_hyperparams.json', 'w') as f:
                                        json.dump(best_hyperparams, f)

                            except Exception as e:
                                print(e)



    # with open('task3_candidates.json', 'r') as f:
    #     candidates = json.load(f)

    # for c in category[1:2]:
    #     for llm in llms[:]:
    #         params = candidates[c][llm]
    #         model = params[0]
    #         seed = params[1]
    #         cut_length = params[2]
    #         data_size = params[3]
    #         f1 = params[4]

    #         command = f"python run_lm.py --dataset {c} --detectLLM {llm} --model {model} --cut_length {cut_length} --data_size {data_size} \
    #         --seed {seed} --save True --task task3"
    #         print(command)
    #         subprocess.run(command, shell=True)
    #         print('should be:', f1)


        
