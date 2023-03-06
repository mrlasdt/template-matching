
from mmocr.apis import init_detector as init_classifier
from mmocr.apis.inference import model_inference
import numpy as np
from .utils import *


class Classifier_SATRN:
    def __init__(self, config, checkpoint, device='cpu'):
        self.model = init_classifier(config, checkpoint,device)

    def inference(self, numpy_image):
        result = model_inference(self.model, numpy_image, batch_mode=True)
        preds_str = [r["text"] for r in result]
        confidence = [r["score"] for r in result]
        return preds_str, confidence
