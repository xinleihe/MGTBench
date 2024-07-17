from ..auto import BaseExperiment, BaseDetector, DetectOutput
from ..methods import MetricBasedDetector, PerturbBasedDetector, SupervisedDetector
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, fields, asdict


class ThresholdExperiment(BaseExperiment):
    _ALLOWED_detector = ['ll', 'rank', 'rankGLTR', 'entropy', 'GPTZero']

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
                print(detector.name, 'is not for', self.__name__)
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
                print(detector.name, 'is not for', self.__name__)
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
                print(detector.name, 'is not for', self.__name__)
                continue
            self.supervise_config.update(kargs)
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
