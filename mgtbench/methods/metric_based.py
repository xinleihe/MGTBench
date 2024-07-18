import numpy as np
import torch
import torch.nn.functional as F
import time
from ..utils import timeit, get_clf_results
from .IntrinsicDim import PHD
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..auto import BaseDetector
from ..loading import load_pretrained
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import warnings
from tqdm import tqdm
# # Under development
# def get_phd(text, base_model, base_tokenizer, DEVICE):
#     # default setting
#     MIN_SUBSAMPLE = 40
#     INTERMEDIATE_POINTS = 7
#     alpha=1.0
#     solver = PHD(alpha=alpha, metric='euclidean', n_points=9)

#     text = text[:200]
#     inputs = base_tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outp = base_model(**inputs)

#     # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
#     mx_points = inputs['input_ids'].shape[1] - 2
#     mn_points = MIN_SUBSAMPLE
#     step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS

#     t1 = time.time()
#     res = solver.fit_transform(outp[0][0].cpu().numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, point_jump=step)
#     print(time.time() - t1, "Seconds")
#     return res

class MetricBasedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained(model_name_or_path, quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        


class LLDetector(MetricBasedDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name,**kargs)


    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                labels = tokenized.input_ids
                result.append( -self.model(**tokenized, labels=labels).loss.item())
        return result if isinstance(text, list) else result[0]
        
 
class RankDetector(MetricBasedDetector):
    def __init__(self,name, **kargs) -> None:
        super().__init__(name,**kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
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
            log = kargs.get("log", False)
            if log:
                ranks = torch.log(ranks)
            result.append(ranks.float().mean().item())
        return result if isinstance(text, list) else result[0]       


class RankGLTRDetector(MetricBasedDetector):
    def __init__(self,name, **kargs) -> None:
        super().__init__(name,**kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
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
            result.append(res)
        return result if isinstance(text, list) else result[0]       


class EntropyDetector(MetricBasedDetector):
    def __init__(self,name, **kargs) -> None:
        super().__init__(name,**kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt").to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
                neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
                result.append( -neg_entropy.sum(-1).mean().item())
        return result if isinstance(text, list) else result[0]

