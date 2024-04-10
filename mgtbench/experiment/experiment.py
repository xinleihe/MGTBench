from ..auto import BaseExperiment, BaseDetector, DetectOutput
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression

class ThresholdExperiment(BaseExperiment):
    _ALLOWED_detector = ['ll', ]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, BaseDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        
    def predict(self):
        predict_list = []
        for detector in self.detector:
            if detector not in self._ALLOWED_detector:
                print(detector.__name__, 'is not for', self.__name__)
            x_train = detector.detect(self.train_text)
            x_test = detector.detect(self.test_text)
            select_train_index = ~np.isnan(x_train)
            select_test_index = ~np.isnan(x_test)
            x_train = x_train[select_train_index]
            y_train = y_train[select_train_index]
            x_test = x_test[select_test_index]
            y_test = y_test[select_test_index]
            clf = LogisticRegression(random_state=0).fit(x_train, y_train)

            y_train_pred = clf.predict(x_train)
            y_train_pred_prob = clf.predict_proba(x_train)
            y_train_pred_prob = [_[1] for _ in y_train_pred_prob]

            y_test_pred = clf.predict(x_test)
            y_test_pred_prob = clf.predict_proba(x_test)
            y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
            predict_list.append({'train_pred':(y_train, y_train_pred, y_train_pred_prob),
                                 'test_pred':(y_test, y_test_pred, y_test_pred_prob)})
        return predict_list

    
