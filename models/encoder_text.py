# Text Encoder return feature
import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name_or_path)
        self.hidden_size = self.text_encoder.config.hidden_size
        
        # whether freeze the parameters of text_encoder
        for param in self.text_encoder.parameters():
            if config.fix_text_encoder_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, text_inputs, image_inputs):
        outputs = self.text_encoder(**text_inputs)
        return outputs.pooler_output, outputs.last_hidden_state