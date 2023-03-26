import numpy as np
import torch
import torch.nn.functional as F
import time
from methods.utils import timeit, get_clf_results


def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317


def get_lls(texts, base_model, base_tokenizer, DEVICE):
    return [get_ll(_, base_model, base_tokenizer, DEVICE) for _ in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

# get the average rank of each observed token sorted by model likelihood


def get_rank_GLTR(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float()
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(len(ranks)):
            if ranks[i] < 10:
                res[0] += 1
            elif ranks[i] < 100:
                res[1] += 1
            elif ranks[i] < 1000:
                res[2] += 1
            else:
                res[3] += 1
        if res.sum() > 0:
            res = res / res.sum()

        return res


# get average entropy of each token in the text
def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


@timeit
def run_threshold_experiment(data, criterion_fn, name):
    torch.manual_seed(0)
    np.random.seed(0)

    # get train data
    train_text = data['train']['text']
    train_label = data['train']['label']
    t1 = time.time()
    train_criterion = [criterion_fn(train_text[idx])
                       for idx in range(len(train_text))]
    x_train = np.array(train_criterion)
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = train_label

    test_text = data['test']['text']
    test_label = data['test']['label']
    test_criterion = [criterion_fn(test_text[idx])
                      for idx in range(len(test_text))]
    x_test = np.array(test_criterion)
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = test_label

    train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test)

    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res

    print(f"{name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(f"{name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return {
        'name': f'{name}_threshold',
        'predictions': {'train': train_criterion, 'test': test_criterion},
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


@timeit
def run_GLTR_experiment(data, criterion_fn, name):
    torch.manual_seed(0)
    np.random.seed(0)

    train_text = data['train']['text']
    train_label = data['train']['label']
    train_criterion = [criterion_fn(train_text[idx])
                       for idx in range(len(train_text))]
    x_train = np.array(train_criterion)
    y_train = train_label

    test_text = data['test']['text']
    test_label = data['test']['label']
    test_criterion = [criterion_fn(test_text[idx])
                      for idx in range(len(test_text))]
    x_test = np.array(test_criterion)
    y_test = test_label

    train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test)

    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res

    print(f"{name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(f"{name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return {
        'name': f'{name}_threshold',
        'predictions': {'train': train_criterion, 'test': test_criterion},
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }
