import torch
import torch.nn as nn
from .encoder_image import ImageEncoder
from .encoder_text import TextEncoder


class HSTECModel(nn.Module):

    def __init__(self, config):
        super(HSTECModel, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.image_encoder_name = config.image_encoder#change point

        self.attention_nhead = 8
        self.attention_dropout = 0.1
        self.proj_hidden_size = 768

        text_hidden_size = self.text_encoder.hidden_size
        self.image_hidden_size = config.image_model_hidden_sizes[config.image_encoder]

        # attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=self.proj_hidden_size,
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

        # residual
        self.text_fc = nn.Sequential(
            nn.Linear(self.proj_hidden_size * 2, self.proj_hidden_size)
        )
        self.image_fc = nn.Sequential(
            nn.Linear(self.proj_hidden_size * 2, self.proj_hidden_size)
        )

        # classification
        self.classifier = nn.Sequential(
            nn.Linear(self.proj_hidden_size, config.num_classes)
        )


    def forward(self, text_inputs, image_inputs):
        text_pooler_output, text_last_hidden_state = self.text_encoder(text_inputs, image_inputs)
        image_pooler_output, image_last_hidden_state = self.image_encoder(text_inputs, image_inputs)

        image_pooler_output = image_pooler_output.view(-1, image_pooler_output.size(1)) # (batch_size, image_hidden_size)
        image_last_hidden_state = image_last_hidden_state.view(image_last_hidden_state.size(0), -1, self.image_hidden_size) # (batch_size, seq_len, image_hidden_size)

        text_last_hidden_state = self.text_proj(text_last_hidden_state) # (batch_size, seq_len, proj_hidden_size)
        image_last_hidden_state = self.image_proj(image_last_hidden_state) # (batch_size, seq_len, proj_hidden_size)


        text_pooler_output = self.text_proj(text_pooler_output) # (batch_size, proj_hidden_size)
        image_pooler_output = self.image_proj(image_pooler_output) # (batch_size, proj_hidden_size)

        # attention
        attention_output = self.attention(
            torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1) # (batch_size, text_seq_len + image_seq_len, proj_hidden_size)
        )
        attention_output = torch.mean(attention_output, dim=1).squeeze(1) # (batch_size, proj_hidden_size)

        # classification
        text_feature = self.text_fc(torch.cat([text_pooler_output, attention_output], dim=1)) # (batch_size, proj_hidden_size)
        image_feature = self.image_fc(torch.cat([image_pooler_output, attention_output], dim=1)) # (batch_size, proj_hidden_size)
        fuse_feature = text_feature + image_feature # (batch_size, proj_hidden_size)
        clf_logits = self.classifier(
            fuse_feature
        )
        return clf_logits

