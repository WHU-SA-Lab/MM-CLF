# Text Encoder return feature
import torch
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):

    def __init__(self, config):
        super(TextClassifier, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(config.text_model_name_or_path)
        # pass through a single layer ffnn
        self.classifier = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, config.num_classes)
        )

        # whether freeze the parameters of text_encoder
        for param in self.text_encoder.parameters():
            if config.fix_text_encoder_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, text_inputs, image_inputs):
        outputs = self.text_encoder(**text_inputs)
        clf_logits = self.classifier(outputs.pooler_output) # (batch_size, num_classes)
        return clf_logits
