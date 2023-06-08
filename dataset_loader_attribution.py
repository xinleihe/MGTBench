import random
import datasets
import tqdm
import pandas as pd
import re

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA', 'SQuAD1', 'NarrativeQA']


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


def load_TruthfulQA(cache_dir):
    f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    mgt_text_list = []
    for detectLLM in ["ChatGPT", "ChatGLM", "Dolly", "ChatGPT-turbo", "GPT4", "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1 or len(mgt_text[i]) >= 2000:
                flag = 0
                break
        if flag:
            res.append([q[i], a_human[i], mgt_text_list[0][i], mgt_text_list[1][i], mgt_text_list[2]
                       [i], mgt_text_list[3][i], mgt_text_list[4][i], mgt_text_list[5][i], c[i]])

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

        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j-1)

    return data_new


def load_SQuAD1(cache_dir):
    f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    mgt_text_list = []
    for detectLLM in ["ChatGPT", "ChatGLM", "Dolly", "ChatGPT-turbo", "GPT4", "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1:
                flag = 0
                break
        if flag:
            res.append([q[i], a_human[i], mgt_text_list[0][i], mgt_text_list[1][i],
                       mgt_text_list[2][i], mgt_text_list[3][i], mgt_text_list[4][i], mgt_text_list[5][i]])

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

        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j-1)
    return data_new


def load_NarrativeQA(cache_dir):
    f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    mgt_text_list = []
    for detectLLM in ["ChatGPT", "ChatGLM", "Dolly", "ChatGPT-turbo", "GPT4", "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1 or len(a_human[i].split()) >= 150:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1 or len(mgt_text[i].split()) >= 150:
                flag = 0
                break
        if flag:
            res.append([q[i], a_human[i], mgt_text_list[0][i], mgt_text_list[1][i],
                       mgt_text_list[2][i], mgt_text_list[3][i], mgt_text_list[4][i], mgt_text_list[5][i]])

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
        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j-1)
    return data_new


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
