from ..auto import BaseExperiment, BaseDetector, DetectOutput
from ..methods import MetricBasedDetector, PerturbBasedDetector, SupervisedDetector, DemasqDetector
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, fields, asdict


class ThresholdExperiment(BaseExperiment):
    _ALLOWED_detector = ['ll', 'rank', 'rankGLTR', 'entropy']

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, MetricBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        
    def predict(self, **config):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for this experiment')
                continue
            print('Predict training data')
            x_train, y_train = self.data_prepare(detector.detect(self.train_text), self.train_label)
            print('Predict testing data')
            x_test, y_test = self.data_prepare(detector.detect(self.test_text), self.test_label)
            print('Run classification for results')
            clf = LogisticRegression(random_state=0).fit(x_train, y_train)
            train_result = self.run_clf(clf, x_train, y_train)
            test_result = self.run_clf(clf, x_test, y_test)
            predict_list.append({'train_pred':train_result,
                                 'test_pred':test_result})
        return predict_list

@dataclass
class PerturbConfig:
    span_length:int = 2
    buffer_size:int = 1
    mask_top_p:float = 1 
    pct_words_masked:float = 0.3
    DEVICE:int = 0
    random_fills:bool = False
    random_fills_tokens:bool = False
    n_perturbation_rounds:int = 1
    n_perturbations:int = 10
    criterion:str = 'd'

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class PerturbExperiment(BaseExperiment):
    '''
    class PerturbConfig:
        span_length:int = 2
        buffer_size:int = 1
        mask_top_p:float = 1 
        pct_words_masked:float = 0.3
        DEVICE:int = 0
        random_fills:bool = False
        random_fills_tokens:bool = False
        n_perturbation_rounds:int = 1
        n_perturbations:int = 10
        criterion:str = 'd'
    '''
    _ALLOWED_detector = ['detectGPT', 'NPR', 'LRR' ]
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, PerturbBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.perturb_config= PerturbConfig()

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            # print(kargs)

            self.perturb_config.update(kargs)
            print('Predict training data')
            x_train, y_train = self.data_prepare(detector.detect(self.train_text, self.train_label, self.perturb_config),self.train_label)
            print('Predict testing data')
            x_test, y_test   = self.data_prepare(detector.detect(self.test_text, self.test_label, self.perturb_config), self.test_label)
            print('Run classification for results')
            clf = LogisticRegression(random_state=0).fit(x_train, y_train)
            train_result = self.run_clf(clf, x_train, y_train)
            test_result = self.run_clf(clf, x_test, y_test)
            predict_list.append({'train_pred':train_result,
                                 'test_pred':test_result})
        return predict_list


@dataclass
class SupervisedConfig:
    need_finetune:bool=False
    need_save:bool=True
    batch_size:int=16
    pos_bit:int=1
    num_labels:int=2
    epochs:int=3
    save_path:str='finetuned/'

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])

class SupervisedExperiment(BaseExperiment):
    '''
    class SupervisedConfig:
        need_finetune:bool=False
        need_save:bool=True
        batch_size:int=16
        pos_bit:int=1
        num_labels:int=2
        epochs:int=3
        save_path:str='finetuned/'
    '''
    _ALLOWED_detector = ['OpenAI-D', 'ConDA', 'ChatGPT-D', "LM-D" ]
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, SupervisedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.supervise_config= SupervisedConfig()

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.supervise_config.update(kargs)
            if self.supervise_config.need_finetune:
                data_train = {'text':self.train_text, 'label':self.train_label}
                detector.finetune(data_train, self.supervise_config)
                print('Fine-tune finished')
            print('Predict training data')
            x_train, y_train = self.data_prepare(detector.detect(self.train_text),self.train_label)
            print('Predict testing data')
            x_test, y_test   = self.data_prepare(detector.detect(self.test_text), self.test_label)
            print('Run classification for results')
            clf = LogisticRegression(random_state=0).fit(x_train, y_train)
            train_result = self.run_clf(clf, x_train, y_train)
            test_result = self.run_clf(clf, x_test, y_test)
            predict_list.append({'train_pred':train_result,
                                 'test_pred':test_result})
        return predict_list


@dataclass
class DemasqConfig:
    need_finetune:bool=True
    need_save:bool=True
    batch_size:int = 1
    save_path:str='model_weight/'
    epoch:int = 12
    
    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class DemasqExperiment(BaseExperiment):
    '''
    class DemasqConfig:
        need_finetune:bool=False
        batch_size:int = 1
        save_path:str='model_weight/'
        epoch:int = 12
    '''
    _ALLOWED_detector = ['demasq']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, DemasqDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.config= DemasqConfig()

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.config.update(kargs)
            if self.config.need_finetune:
                data_train = {'text':self.train_text, 'label':self.train_label}
                detector.finetune(data_train, self.config)
            
            logits = detector.detect(self.train_text)
            preds = [1 if logit>0.5 else 0 for logit in logits]
            logits_t = detector.detect(self.test_text)
            preds_t = [1 if logit>0.5 else 0 for logit in logits_t]
            predict_list.append({'train_pred':(self.train_label, preds, logits),
                                 'test_pred':(self.test_label, preds_t, logits_t)})
        return predict_list

class GPTZeroExperiment(BaseExperiment):
    _ALLOWED_detector = ['GPTZero']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, DemasqDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for _ALLOWED_detector')
                continue
            
            logits = detector.detect(self.train_text)
            preds = [1 if logit>0.5 else 0 for logit in logits]
            logits_t = detector.detect(self.test_text)
            preds_t = [1 if logit>0.5 else 0 for logit in logits_t]
            predict_list.append({'train_pred':(self.train_label, preds, logits),
                                 'test_pred':(self.test_label, preds_t, logits_t)})
        return predict_list
