import random
import datasets
import tqdm
import pandas as pd
import re
import os
import json

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA', 'SQuAD1', 'NarrativeQA', "Essay", "Reuters", "WP"]


def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def process_text_truthfulqa_adv(text):

    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def check_period(texts):
    for i in range(len(texts)):
        if texts[i][-1] != ".":
            texts[i] += "."
    return texts


def load_TruthfulQA(detectLLM):
    f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_human = check_period(a_human)
    print(a_human)
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(
            a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
                a_chat[i]) < 2000:
            res.append([q[i], a_human[i], a_chat[i], c[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)

        data_new[data_partition]['category'].append(res[index_list[i]][3])
        data_new[data_partition]['category'].append(res[index_list[i]][3])

    return data_new


def load_SQuAD1(detectLLM):
    f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            a_human[i] = check_period(a_human[i])
            res.append([q[i], a_human[i], a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'

        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load_NarrativeQA(detectLLM):
    f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(
            a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
            a_human[i].split()) < 150 and len(
                a_chat[i].split()) < 150:
            a_human[i] = check_period(a_human[i])
            res.append([q[i], a_human[i], a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load(name, detectLLM, task='task2', cut_length=3000, disable=False, seed=0):

    if name in ['TruthfulQA', 'SQuAD1', 'NarrativeQA']:
        load_fn = globals()[f'load_{name}']
        return load_fn(detectLLM)
    elif name in ["Essay", "Reuters", "WP"]:

        f = pd.read_csv(f"datasets/{name}_LLMs.csv")
        a_human = f["human"].tolist()
        a_chat = f[f'{detectLLM}'].fillna("").tolist()

        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=disable):
            if i < total_num * 0.8:
                data_partition = 'train'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)

        return data_new
    
    elif name in ['Physics','Medicine','Biology','Electrical_engineering','Computer_science','Literature','History','Education','Art','Law','Management','Philosophy','Economy','Math','Statistics','Chemistry']:
        human_data = datasets.load_dataset("/data1/zzy/MGT-human")
        subject_human_data = human_data[name]
        if task == 'task1':
            task_path = '/data1/zzy/AIGen-TASK1'
        elif task == 'task2' or task == 'task2_gen':
            task_path = '/data1/zzy/AIGen-TASK2'
        elif task == 'task3':
            task_path = '/data1/zzy/AIGen-TASK3'
        else:
            raise ValueError(f'Unknown task {task}')

        # mgt data
        files = os.listdir(task_path)
        files = [f for f in files if f.endswith('.json')]
        files = [f for f in files if (f.split('.')[0].endswith(detectLLM) and f.startswith(name))]
        mgt_data = None
        for f in files:
            with open(f'{task_path}/{f}', 'r') as f:
                js = json.load(f)
                if mgt_data is None:
                    mgt_data = js
                else:
                    mgt_data += js
    
        # data mix up
        smaller_len = min([len(subject_human_data), len(mgt_data)])
        subject_human_data = subject_human_data.shuffle(seed)
        all_data = []
        for i in range(smaller_len): # 50:50
            all_data.append({'text': subject_human_data[i]['text'], 'label': 0})

            if task == 'task2':
                ai_complete = mgt_data[i]['prompt'].splitlines()[-2].strip('\"') + mgt_data[i]['generated_text'] # concat generated text with first half 
                all_data.append({'text': ai_complete, 'label': 1})
            elif task == 'task2_gen':
                all_data.append({'text': mgt_data[i]['generated_text'], 'label': 1})
            elif task == 'task3':
                all_data.append({'text': mgt_data[i]['generated_text'], 'label': 1})
            else:
                raise ValueError(f'Unknown task {task}')

        # temp: trim text to 
        for i in range(len(all_data)):
            all_data[i]['text'] = all_data[i]['text'][:cut_length]

        index_list = list(range(len(all_data)))
        random.shuffle(index_list)

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            }

        }

        total_num = len(all_data)
        for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=disable):
            if i < total_num * 0.8:
                data_partition = 'train'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(all_data[index_list[i]]['text']))
            data_new[data_partition]['label'].append(all_data[index_list[i]]['label'])
        return data_new

    else:
        raise ValueError(f'Unknown dataset {name}')
