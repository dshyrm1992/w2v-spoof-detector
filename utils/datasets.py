import torch
import librosa
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


def load_signal(path, target_sr=16_000):

    sig, sr = librosa.load(path)

    # handle multichannel signals
    if len(sig.shape) == 2:
        sig = sig.mean(axis=1)

    # resample if needed
    if sr != target_sr:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)

    return sig


class BaseSpoofDataset(Dataset):

    def __init__(
        self,
        meta_path,
        data_path,
        sr=16_000,
        feature_extractor=None
    ):

        self.sr = sr
        self.feature_extractor = feature_extractor
        self.data_path = data_path

        self.meta = pd.read_csv(meta_path, header=None, sep=' ', names=['speaker', 'sample_id', 'system', '-', 'label'])

    def __len__(self):

        return len(self.meta)

    def extract_features(self, sig, **kwargs):

        if self.feature_extractor is not None:
            return self.feature_extractor(sig, **kwargs)['input_values'][0]
        else:
            return sig

    def __getitem__(self, idx):

        sample_meta = self.meta.iloc[idx]

        label = 1 if sample_meta.label == 'bonafide' else 0

        sig = load_signal(Path(self.data_path) / f'{sample_meta.sample_id}.flac', target_sr=self.sr)
        sig = self.extract_features(sig, sampling_rate=self.sr)

        return {
            'sig': torch.Tensor(sig),
            'label': torch.LongTensor([label]),
            'speaker': sample_meta.speaker
        }

