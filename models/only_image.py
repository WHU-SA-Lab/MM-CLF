# Vit return feature and hidden_state
import torch
import torch.nn as nn
from transformers import AutoModel


class ImageClassifier(nn.Module):

    def __init__(self, config):
        super(ImageClassifier, self).__init__()

        self.image_encoder = AutoModel.from_pretrained(config.image_model_name_or_path)
        # pass through a single layer ffnn
        hidden_size = config.image_model_hidden_sizes[config.image_encoder]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, config.num_classes)
        )

        # whether freeze the parameters of text_encoder
        for param in self.image_encoder.parameters():
            if config.fix_image_encoder_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, text_inputs, image_inputs):
        outputs = self.image_encoder(**image_inputs)
        clf_logits = self.classifier(outputs.pooler_output.view(-1, outputs.pooler_output.size(1)))
        return clf_logits