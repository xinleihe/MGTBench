from abc import ABC, abstractmethod
from ..loading import load_pretrained


DETECTOR_MAPPING = {
    'gptzero' : 'mgtbench.methods.GPTZeroAPI',
    'll' : 'mgtbench.methods.metric_based.LLDetector'
}

class AutoDetector:
    _detector_mapping = DETECTOR_MAPPING

    @classmethod
    def from_metric_name(cls, metric_name, *args, **kargs):
        if metric_name not in cls._detector_mapping:
            raise ValueError(f"Unrecognized metric name: {metric_name}")

        metric_class_path = cls._detector_mapping[metric_name]
        module_name, class_name = metric_class_path.rsplit('.', 1)
        
        # Dynamically import the module and retrieve the class
        metric_module = __import__(module_name, fromlist=[class_name])
        metric_class = getattr(metric_module, class_name)
        return metric_class(*args, **kargs)
    

class BaseDetector(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect(self, **kargs):
        raise NotImplementedError('Invalid detector, implement detect first.')


class MetricBasedDetector(BaseDetector):
    def __init__(self,  **kargs) -> None:
        super().__init__()
        model_name_or_path = kargs.get('model_name_or_path', None)
        if not model_name_or_path:
            raise ValueError('You should pass the model_name_or_path, but',model_name_or_path, 'is given')
        quantitize_bit = kargs.get('load_in_k_bit', None)
        self.model, self.tokenizer = load_pretrained(model_name_or_path, quantitize_bit)


class ModelBasedDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__()