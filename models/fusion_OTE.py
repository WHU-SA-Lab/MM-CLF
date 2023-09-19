import torch
import torch.nn as nn
from .encoder_image import ImageEncoder
from .encoder_text import TextEncoder


class OTEModel(nn.Module):

    def __init__(self, config):
        super(OTEModel, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)

        self.attention_nhead = 8
        self.attention_dropout = 0.1
        self.proj_hidden_size = 768

        text_hidden_size = self.text_encoder.hidden_size
        image_hidden_size = config.image_model_hidden_sizes[config.image_encoder]

        # attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=self.proj_hidden_size * 2,
            nhead=self.attention_nhead, 
            dropout=self.attention_dropout,
            batch_first=True
        )

        # projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, self.proj_hidden_size)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_hidden_size, self.proj_hidden_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.proj_hidden_size * 2, config.num_classes)
        )


    def forward(self, text_inputs, image_inputs):
        text_pooler_output, text_last_hidden_state = self.text_encoder(text_inputs, image_inputs)
        image_pooler_output, image_last_hidden_state = self.image_encoder(text_inputs, image_inputs)

        image_pooler_output = image_pooler_output.view(-1, image_pooler_output.size(1)) # (batch_size, image_hidden_size)

        # attention
        text_pooler_output = self.text_proj(text_pooler_output)
        image_pooler_output = self.image_proj(image_pooler_output)
        attention_output = self.attention(
            torch.cat([text_pooler_output, image_pooler_output], dim=1).unsqueeze(1) # (batch_size, 1, proj_hidden_size * 2)
        ) # (batch_size, 1, proj_hidden_size * 2)

        # classification
        clf_logits = self.classifier(
            attention_output.squeeze(1)
        )
        return clf_logits
        