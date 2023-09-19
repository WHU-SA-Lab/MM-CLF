# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import warnings
# warnings.filterwarnings("ignore")
# import sys
# sys.path.append('./utils')
# sys.path.append('./utils/APIs')

# import torch

# import argparse
# from Config import config, model_path_dict
# from utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file
# from Models import ConcatModel, CombineModel, CMACModel, HSTECModel, OTEModel
# from utils.DataProcess import Processor
# from Trainer import Trainer
# import pickle


import os
import argparse
from config import base_config
from utils import setup_logger, read_data, save_cache_data, read_cache_data, setup_processor
# from models import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoImageProcessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from train import Trainer
import open_clip


# command line args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='dataset name, i.e. douyin_fake_comments', type=str)
parser.add_argument('--do_train', action='store_true', help='train model on train set')
parser.add_argument('--do_test', action='store_true', help='evaluate model on test set')
parser.add_argument('--text_encoder', default='bert', help='encoder for text, i.e. bert, roberta', type=str)
parser.add_argument('--image_encoder', default='resnet', help='encoder for image, i.e. resnet, vit', type=str)
parser.add_argument('--text_max_seq_len', default=256, help='max sequence length for text', type=int)
parser.add_argument('--image_size', default=224, help='image size for image encoder', type=int)
parser.add_argument('--image_hidden_size', default=64, help='image hidden size for CMAC, HSTEC and OTE', type=int)
parser.add_argument('--image_seq_len', default=64, help='image sequence length for CMAC, HSTEC and OTE', type=int)
parser.add_argument('--fuse_model_type', default='concat', help='model for fusing text and image modalities, i.e. concat, combine, CMAC, HSTEC, OTE', type=str)
parser.add_argument('--lr', default=5e-5, help='set learning rate', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='set weight decay', type=float)
parser.add_argument('--epoch', default=1, help='set training epoch', type=int)
parser.add_argument('--load_model_path', default=None, help='path to load trained model', type=str)
parser.add_argument('--only_text', action='store_true', help='only use text to predict')
parser.add_argument('--only_image', action='store_true', help='only use image to predict')
parser.add_argument('--cuda_device', default=0, help='set cuda device', type=int)
parser.add_argument('--use_cache', action='store_true', help='use cache data or create if no cache data found')
args = parser.parse_args()

# only_text and only_image cannot be True at the same time
if args.only_text and args.only_image:
    raise ValueError('only_text and only_image cannot be True at the same time!')
if args.only_text:
    args.image_encoder = "none"
    args.fuse_model_type = "only_text"
if args.only_image:
    args.text_encoder = "none"
    args.fuse_model_type = "only_image"

# update base_config with command line args
config = base_config()
config.__update__(vars(args))
config.device = torch.device("cuda:" + str(config.cuda_device) if torch.cuda.is_available() else "cpu")

# set up logger
logger = setup_logger(config)
logger.info('Text Encoder: {}, Image Encoder: {}, Modality Fusion: {}'.format(config.text_encoder, config.image_encoder, config.fuse_model_type))
# iterate through config and log
logger.info('')
logger.info('Config:') 
for k, v in vars(config).items():
    logger.info(f'{k}: {v}')


logger.info('')
logger.info(f'Loading data with use_cache: {config.use_cache}...')
if config.use_cache:
    cache_path = os.path.join(config.data_path, config.dataset + "_dump")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(os.path.join(cache_path, "train_data.pkl")) or not os.path.exists(os.path.join(cache_path, "test_data.pkl")):
        logger.info('Cache data not found, creating cache data...')
        tokenizer, image_processor = setup_processor(config)
        train_data = read_data(config.train_data_path, config.image_data_path, tokenizer, image_processor, config)
        test_data = read_data(config.test_data_path, config.image_data_path, tokenizer, image_processor, config)
        save_cache_data(train_data, os.path.join(config.data_path, config.dataset + "_dump", "train_data.pkl"))
        save_cache_data(test_data, os.path.join(config.data_path, config.dataset + "_dump", "test_data.pkl"))
    logger.info('Loading cache data...')
    train_data = read_cache_data(os.path.join(config.data_path, config.dataset + "_dump", "train_data.pkl"))
    test_data = read_cache_data(os.path.join(config.data_path, config.dataset + "_dump", "test_data.pkl"))
else:
    logger.info('Processing data...')
    tokenizer, image_processor = setup_processor(config)
    train_data = read_data(config.train_data_path, config.image_data_path, tokenizer, image_processor, config)
    test_data = read_data(config.test_data_path, config.image_data_path, tokenizer, image_processor, config)
    save_cache_data(train_data, os.path.join(config.data_path, config.dataset + "_dump", "train_data.pkl"))
    save_cache_data(test_data, os.path.join(config.data_path, config.dataset + "_dump", "test_data.pkl"))    


# create dataloader from data
train_loader = DataLoader(train_data, **config.train_params)
test_loader = DataLoader(test_data, **config.test_params)

logger.info('')
logger.info(f"Train data data: {len(train_data):>6}, Train loader size: {len(train_loader):>5}")
logger.info(f" Test data data: {len(test_data):>6},  Test loader size: {len(test_loader):>5}")

# Initilaztion
model = config.model_mapping[config.fuse_model_type](config)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_loader) * config.epoch
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

trainer = Trainer(model, optimizer, scheduler, config, logger)

best_metric = 0
for epoch in range(config.epoch):
    if config.do_train:
        logger.info('')
        logger.info(f'Epoch: {epoch + 1}')
        label_ls, pred_ls, train_results_dict = trainer.train(train_loader)
        label_ls, pred_ls, test_results_dict = trainer.test(test_loader)
        metric = test_results_dict['macro_f1_score']
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), os.path.join(config.output_path, 'best_model.pth'))
            logger.info(f'update best model, macro_f1_score: {best_metric}')
if config.do_test:
    logger.info('')
    logger.info(f'Evaluating the best model on test set')
    model.load_state_dict(torch.load(os.path.join(config.output_path, 'best_model.pth')))
    label_ls, pred_ls, test_results_dict = trainer.test(test_loader)
