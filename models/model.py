import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Model, HubertModel, Wav2Vec2FeatureExtractor
from speechbrain.lobes.models.huggingface_wav2vec import logger


class TransformersEncoder(nn.Module):

    encoders = {
        'wav2vec': Wav2Vec2Model,
        'hubert': HubertModel
    }

    def __init__(
        self,
        model_type,
        encoder_params=None,
        freeze_feature_extractor=True,
        input_layer_norm=True,
        output_norm=True,
        apply_attention_masking=None,
        freeze=True
    ):

        super().__init__()

        if 'wav2vec' in model_type:
            encoder = self.encoders['wav2vec']
        elif 'hubert' in model_type:
            encoder = self.encoders['hubert']
        else:
            raise ValueError(f'Unsupported task {model_type}'
                             f'available options are {self.encoders.keys()}')

        self.input_layer_norm = input_layer_norm
        self.apply_attention_masking = apply_attention_masking

        self.encoder = encoder.from_pretrained(
            model_type,
            **({} if encoder_params is None else encoder_params)
        )

        self.freeze_feature_extractor = freeze_feature_extractor

        # feature extractor is needed only to know whether padding mask needed
        self.feature_extractor = Wav2Vec2FeatureExtractor().from_pretrained(model_type)

        # freezing is copied from SB class to match the same interface
        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is frozen."
            )
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder.train()
            if self.freeze_feature_extractor:
                self.encoder.feature_extractor._freeze_parameters()

    def extract_features(self, wav, padding_mask=None):

        # ignore padding mask if model's processor do not expect it
        if not self.is_padding_mask_needed:
            padding_mask = None

        if self.freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        if self.input_layer_norm:
            wav = F.layer_norm(wav, wav.shape)

        out = self.encoder(wav, attention_mask=padding_mask)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def forward(self, wav, padding_mask=None):

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav, padding_mask).detach()

        return self.extract_features(wav, padding_mask)

    @property
    def is_padding_mask_needed(self):

        if self.apply_attention_masking is None:
            return self.feature_extractor.return_attention_mask
        else:
            return self.apply_attention_masking

    @property
    def output_size(self):

        return self.encoder.config.output_hidden_size


class MaxPoolClassifier(nn.Module):

    def __init__(self, dim, dropout=0):

        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, x, classify=True):

        x = self.pool(torch.swapaxes(x, 1, 2)).squeeze(-1)
        x = self.dropout(x)

        if classify:
            return self.classifier(x)
        else:
            return x


class BinaryW2VClassifier(nn.Module):

    def __init__(self, encoder, classifier_do=0):

        super().__init__()

        self.encoder = encoder
        self.classifier = MaxPoolClassifier(
            self.encoder.output_size,
            classifier_do
        )

    def forward(self, wav, padding_mask=None):

        features = self.encoder(wav, padding_mask)

        return self.classifier(features)
