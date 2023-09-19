# Vit return feature and hidden_state
import torch
import torch.nn as nn
from transformers import AutoModel
import config
import open_clip

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()

        if config.image_encoder in config.clip_models:
            self.use_clip = True
            self.image_encoder = open_clip.create_model('ViT-L-14', pretrained='openai')
        else:
            self.use_clip = False
            self.image_encoder = AutoModel.from_pretrained(config.image_model_name_or_path)
        
        # whether freeze the parameters of image_encoder
        for param in self.image_encoder.parameters():
            if config.fix_image_encoder_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, text_inputs, image_inputs):

        if self.use_clip: 
            outputs = self.image_encoder(image_inputs)
            return outputs[0], None # (batch_size, image_hidden_size)
        else:    
            outputs = self.image_encoder(**image_inputs)
            return outputs.pooler_output, outputs.last_hidden_state # (batch_size, image_hidden_size), (batch_size, seq_len, image_hidden_size)