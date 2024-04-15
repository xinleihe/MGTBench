from ..auto import BaseExperiment, BaseDetector, DetectOutput
from ..methods import MetricBasedDetector, PerturbBasedDetector
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass


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
    DEVICE = 0
    random_fills = False
    random_fills_tokens = False
    n_perturbation_rounds = 1



class PerturbExperiment(BaseExperiment):
    _ALLOWED_detector = ['detectGPT', 'NPR', 'LRR' ]
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, PerturbBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.perturb_config= PerturbConfig()
        self.n_perturbations = 10

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for', self.__name__)
                continue
            for k, v in kargs.items():
                if k in self.perturb_config:
                    self.perturb_config.k = v   
            detect_config = {
                'n_perturbations':self.n_perturbations, 
                'perturb_config':self.perturb_config 
            }
            print('Predict training data')
            x_train, y_train = self.data_prepare(detector.detect(self.train_text, self.train_label, detect_config),self.train_label)
            print('Predict testing data')
            x_test, y_test   = self.data_prepare(detector.detect(self.test_text, self.test_label, detect_config), self.test_label)
            print('Run classification for results')
            clf = LogisticRegression(random_state=0).fit(x_train, y_train)
            train_result = self.run_clf(clf, x_train, y_train)
            test_result = self.run_clf(clf, x_test, y_test)
            predict_list.append({'train_pred':train_result,
                                 'test_pred':test_result})
        return predict_list
