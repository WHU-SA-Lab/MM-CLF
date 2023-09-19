import os
import json
from models import *


class base_config(object):

    def __init__(self):
        self.root_path = os.getcwd()
        self.data_path = os.path.join(self.root_path, 'data')
        # the path where datasets are stored
        self.output_path = os.path.join(self.root_path, 'output')
        # the path where trained model and log will be saved

        # dataloader params
        self.train_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 4}
        self.val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 4}
        self.test_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 4}

        # model params
        self.model_path_dict = {
            "bert": "/workspace/model/chinese-bert-wwm-ext",
            "roberta": "/workspace/model/chinese-roberta-wwm-ext",
            "resnet": "/workspace/model/resnet-50",
            "vit": "/workspace/model/vit-base-patch16-224",
            "taiyi": "/workspace/model/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese",
            "none": None
        }

        self.image_model_hidden_sizes = {
            "resnet": 2048,
            "vit": 768,
            "taiyi": 768,
            "none": None
        }

        self.fix_text_encoder_params = False
        self.fix_image_encoder_params = False

        # model_mapping
        self.model_mapping = {
            "only_text": TextClassifier,
            "only_image": ImageClassifier,
            "concat": ConcatModel,
            "combine": CombineModel,
            "OTE": OTEModel,
            "CMAC": CMACModel,
            "HSTEC": HSTECModel
        }

        self.clip_models = ['taiyi']


    def __update__(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.train_data_path = os.path.join(self.data_path, self.dataset, "train.jsonl")
        self.test_data_path = os.path.join(self.data_path, self.dataset, "test.jsonl")
        self.image_data_path = os.path.join(self.data_path, self.dataset, "images")
        self.output_path = os.path.join(self.output_path, self.dataset,  self.fuse_model_type + "_" + self.text_encoder + "_" + self.image_encoder)
        self.num_classes = json.load(open(os.path.join(self.data_path, self.dataset, 'config.json'), 'r'))['num_classes']
        self.text_model_name_or_path = self.model_path_dict[self.text_encoder]
        self.image_model_name_or_path = self.model_path_dict[self.image_encoder]

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    