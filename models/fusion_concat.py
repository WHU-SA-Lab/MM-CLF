import torch
import torch.nn as nn
from .encoder_image import ImageEncoder
from .encoder_text import TextEncoder


class ConcatModel(nn.Module):

    def __init__(self, config):
        super(ConcatModel, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)

        text_hidden_size = self.text_encoder.hidden_size
        image_hidden_size = config.image_model_hidden_sizes[config.image_encoder]
        
        # a single layer ffnn
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + image_hidden_size, config.num_classes)
        )
        

    def forward(self, text_inputs, image_inputs):
        text_pooler_output, text_last_hidden_state = self.text_encoder(text_inputs, image_inputs)
        image_pooler_output, image_last_hidden_state = self.image_encoder(text_inputs, image_inputs)

        image_pooler_output = image_pooler_output.view(-1, image_pooler_output.size(1)) # (batch_size, image_hidden_size)

        clf_logits = self.classifier(
            torch.cat([text_pooler_output, image_pooler_output], dim=1)
        )
        return clf_logits