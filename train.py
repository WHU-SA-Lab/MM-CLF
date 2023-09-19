from tqdm import tqdm
import torch
from torch.optim import AdamW
from utils import clf_results
import config
    
class Trainer():
     
    def __init__(self, model, optimizer, scheduler, config, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.image_encoder = config.image_encoder
        self.logger = logger
        self.device = config.device
        self.model.to(self.device)
    
    def train(self, train_loader, k=50):
        self.model = self.model.train()
        
        total_loss = 0.0
        label_ls, pred_ls = [], []
        step = 0
        self.logger.info('Training...')
        for batch in tqdm(train_loader, desc='----- [Training] '):
            text_inputs = batch['input']['text']
            image_inputs = batch['input']['image']

            text_inputs = {k: v.squeeze(1).to(self.device) for k, v in text_inputs.items()} # (batch_size, 1, seq_len) -> (batch_size, seq_len) i.e. (batch_size, 512)
            
            if self.image_encoder in self.config.clip_models:
                image_inputs = image_inputs.to(self.device) # image_inputs is tensor when using CLIP models, while it is dict when using huggingface transformers models
            else:
                image_inputs = {k: v[0].to(self.device) for k, v in image_inputs.items()} # (batch_size, RGB_channel, image_size, image_size) i.e. (batch_size, 3, 224, 224)

            labels = batch['label'].to(self.device)
        
            outputs = self.model(text_inputs, image_inputs)
            loss = self.compute_loss(outputs, labels)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Append the labels and predictions for metric computation
            label_ls.extend(labels.detach().cpu().numpy())
            pred_ls.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            
            # Log the training loss every k steps
            if (step + 1) % k == 0:
                avg_loss = total_loss / k
                self.logger.info(f'Step: {step + 1}, Average Training Loss: {avg_loss:.4f}')
                total_loss = 0.0
            
            step += 1
            
        clf_results_dict = clf_results(label_ls, pred_ls, labels=[i for i in range(self.config.num_classes)])
        self.logger.info('micro precision: {}, recall: {}, f1_score: {}'.format(clf_results_dict['micro_precision'], clf_results_dict['micro_recall'], clf_results_dict['micro_f1_score']))
        self.logger.info('macro precision: {}, recall: {}, f1_score: {}'.format(clf_results_dict['macro_precision'], clf_results_dict['macro_recall'], clf_results_dict['macro_f1_score']))

        return label_ls, pred_ls, clf_results_dict
    
    def compute_loss(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)
    
    def test(self, test_loader):
        # clear cuda memory
        torch.cuda.empty_cache()
        # set model to evaluation mode
        self.model = self.model.eval()
        # create a no_grad context environment
        with torch.no_grad():
            label_ls, pred_ls = [], []
            for batch in tqdm(test_loader, desc='----- [Testing] '):
                text_inputs = batch['input']['text']
                image_inputs = batch['input']['image']
                text_inputs = {k: v.squeeze(1).to(self.device) for k, v in text_inputs.items()} # (batch_size, 1, seq_len) -> (batch_size, seq_len) i.e. (batch_size, 512)
                if self.image_encoder == "taiyi":
                    image_inputs = image_inputs.to(self.device)
                else:
                    image_inputs = {k: v[0].to(self.device) for k, v in image_inputs.items()} # (batch_size, RGB_channel, image_size, image_size) i.e. (batch_size, 3, 224, 224)

                labels = batch['label'].to(self.device)
                
                outputs = self.model(text_inputs, image_inputs)
                
                # Append the labels and predictions for metric computation
                label_ls.extend(labels.detach().cpu().numpy())
                pred_ls.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

        clf_results_dict = clf_results(label_ls, pred_ls, labels=[i for i in range(self.config.num_classes)])
        self.logger.info('Evaluating...')
        self.logger.info('micro precision: {}, recall: {}, f1_score: {}'.format(clf_results_dict['micro_precision'], clf_results_dict['micro_recall'], clf_results_dict['micro_f1_score']))
        self.logger.info('macro precision: {}, recall: {}, f1_score: {}'.format(clf_results_dict['macro_precision'], clf_results_dict['macro_recall'], clf_results_dict['macro_f1_score']))

        return label_ls, pred_ls, clf_results_dict
