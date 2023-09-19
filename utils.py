import os
import logging
from tqdm import tqdm
import json
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import pickle
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoImageProcessor
import open_clip


# Set up logger
def setup_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(config.output_path, config.fuse_model_type + "_" + config.text_encoder + "_" + config.image_encoder + ".log")
    with open(log_file, 'w') as f:
        pass # clear existing log file
    
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


# read data from file
def read_data(data_path, image_dir, tokenizer, image_processor, config):

    # mean and std of imagenet dataset
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    # image transform
    image_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        # transforms.Resize(config.image_size),
        # transforms.CenterCrop(config.image_size),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=image_mean,
        #     std=image_std)
        ])
    
    # mapping label str to int
    label2int = {str(i): i for i in range(config.num_classes)}

    data = []
    with open(data_path) as f:
        for d in tqdm(f, desc="-" * 10 + f' [Loading raw data from {data_path}] ' + "-" * 10):

            # read and encode text
            d = json.loads(d)
            id, label, text = d['id'], d['label'], d['text']
            encoded_text = tokenizer.encode_plus(
                text,
                max_length=config.text_max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # read and encode image
            image_path = os.path.join(image_dir, (id + '.jpg'))
            image = Image.open(image_path)
            image.load()
            processed_image = image_processor(image)

            # convert label str to tensor
            label = label2int[label]
            label = torch.tensor(label, dtype=torch.long)

            example = {
                'label': label,
                'input': {"image": processed_image, "text": encoded_text}
            }
            data.append(example)
        f.close()

    return data


def save_cache_data(data, cache_path):
    # create cache dir if not exists
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)


def read_cache_data(cache_path):
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data


def clf_results(label_ls, pred_ls, labels):
    clf_results = classification_report(label_ls, pred_ls, labels=labels, output_dict=True)
    micro_precision = clf_results['weighted avg']['precision']
    micro_recall = clf_results['weighted avg']['recall']
    micro_f1_score = clf_results['weighted avg']['f1-score']
    macro_precision = clf_results['macro avg']['precision']
    macro_recall = clf_results['macro avg']['recall']
    macro_f1_score = clf_results['macro avg']['f1-score']
    clf_results_dict = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1_score': micro_f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score
    }
    clf_results_dict = {k: round(v*100, 4) for k, v in clf_results_dict.items()}
    return clf_results_dict

def setup_processor(config):
    
    # use bert as default tokenizer if only_image is set to True
    tokenizer_path = config.text_model_name_or_path if config.text_model_name_or_path is not None else config.model_path_dict['bert']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # use resnet as default image processor if only_text is set to True
    image_processor_path = config.image_model_name_or_path if config.image_model_name_or_path is not None else config.model_path_dict['resnet']
    if config.image_encoder == "taiyi":
        clip_model, _, image_processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    else:
        image_processor = AutoImageProcessor.from_pretrained(image_processor_path)

    return tokenizer, image_processor
