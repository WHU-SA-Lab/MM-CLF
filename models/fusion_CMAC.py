import torch
import torch.nn as nn
from .encoder_image import ImageEncoder
from .encoder_text import TextEncoder


class CMACModel(nn.Module):

    def __init__(self, config):
        super(CMACModel, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.image_encoder_name = config.image_encoder#change point

        self.attention_nhead = 8
        self.attention_dropout = 0.1
        self.proj_hidden_size = 768

        self.text_hidden_size = self.text_encoder.hidden_size
        self.image_hidden_size = config.image_model_hidden_sizes[config.image_encoder]

        # attention
        self.text_image_attention = nn.MultiheadAttention(
            embed_dim=self.proj_hidden_size,
            num_heads=self.attention_nhead,
            dropout=self.attention_dropout,
            batch_first=True
        )
        self.image_text_attention = nn.MultiheadAttention(
            embed_dim=self.proj_hidden_size,
            num_heads=self.attention_nhead,
            dropout=self.attention_dropout,
            batch_first=True
        )

        # projection
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.proj_hidden_size)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.proj_hidden_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.proj_hidden_size * 2, config.num_classes)
        )


    def forward(self, text_inputs, image_inputs):
        text_pooler_output, text_last_hidden_state = self.text_encoder(text_inputs, image_inputs)
        image_pooler_output, image_last_hidden_state = self.image_encoder(text_inputs, image_inputs)

        image_last_hidden_state = image_last_hidden_state.view(image_last_hidden_state.size(0), -1, self.image_hidden_size) # (batch_size, seq_len, image_hidden_size)

        text_last_hidden_state = self.text_proj(text_last_hidden_state) # (batch_size, seq_len, proj_hidden_size)
        image_last_hidden_state = self.image_proj(image_last_hidden_state) # (batch_size, seq_len, proj_hidden_size)

        # attention
        text_image_attention_output, _ = self.text_image_attention(
            text_last_hidden_state, image_last_hidden_state, image_last_hidden_state
        ) # (batch_size, seq_len, proj_hidden_size)
        image_text_attention_output, _ = self.image_text_attention(
            image_last_hidden_state, text_last_hidden_state, text_last_hidden_state
        ) # (batch_size, seq_len, proj_hidden_size)
        text_image_attention_output = torch.mean(text_image_attention_output, dim=1).squeeze(1) # (batch_size, proj_hidden_size)
        image_text_attention_output = torch.mean(image_text_attention_output, dim=1).squeeze(1) # (batch_size, proj_hidden_size)

        clf_logits = self.classifier(
            torch.cat([text_image_attention_output, image_text_attention_output], dim=1)
        )

        return clf_logits
