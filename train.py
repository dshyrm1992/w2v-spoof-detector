import sys

from torch.utils.data import DataLoader
from torch.nn import functional as F
import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from hyperpyyaml import load_hyperpyyaml

from utils import BaseSpoofDataset, get_padding_mask


class SpoofBrain(sb.Brain):

    def compute_forward(self, batch, stage):

        wavs, durations = batch.sig

        if self.modules.model.encoder.is_padding_mask_needed:
            padding_mask = get_padding_mask(wavs, durations).to(self.device)
        else:
            padding_mask = None

        return self.modules.model(
            wav=wavs.to(self.device),
            padding_mask=padding_mask,
        )

    def compute_objectives(self, predictions, batch, stage):

        return F.binary_cross_entropy_with_logits(predictions, batch.label.data.float().to(self.device))

    def fit_batch(self, batch):

        loss = super().fit_batch(batch)

        if hasattr(self.hparams, "lr_scheduler"):
            self.hparams.lr_scheduler(self.optimizer)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):

        if stage == sb.Stage.VALID:

            print(f'Test loss: {stage_loss}')

            self.checkpointer.save_and_keep_only(
                meta={
                    'epoch': epoch,
                    'valid_loss': stage_loss,
                },
                num_to_keep=self.hparams.train_params['checkpoints_to_keep'],
                min_keys=['valid_loss']
            )


def init_data_loaders(params):

    feature_extractor = params.get('feature_extractor')
    if feature_extractor is not None:
        feature_extractor = feature_extractor.from_pretrained(params['model_type'])

    train_loader = DataLoader(
        dataset=BaseSpoofDataset(
            meta_path=params['data']['train']['meta'],
            data_path=params['data']['train']['audio'],
            sr=params['sr'],
            feature_extractor=feature_extractor
        ),
        batch_size=params['train_params']['batch_size'],
        num_workers=params['train_params']['num_workers'],
        collate_fn=PaddedBatch,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=BaseSpoofDataset(
            meta_path=params['data']['test']['meta'],
            data_path=params['data']['test']['audio'],
            sr=params['sr'],
            feature_extractor=feature_extractor
        ),
        batch_size=params['train_params']['batch_size'],
        num_workers=params['train_params']['num_workers'],
        collate_fn=PaddedBatch,
        drop_last=False,
    )

    return train_loader, test_loader


if __name__ == '__main__':

    params_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    params = load_hyperpyyaml(open(params_file), overrides)
    run_opts['auto_mix_prec'] = params['auto_mix_prec']

    sb.create_experiment_directory(
        experiment_directory=params['output_folder'],
        hyperparams_to_save=params_file,
        overrides=overrides
    )

    train_loader, test_loader = init_data_loaders(params)

    params['checkpointer'].recover_if_possible()

    SpoofBrain(
        modules={"model": params['model']},
        opt_class=lambda x: params['optimizer'](x),
        hparams=params,
        run_opts=run_opts,
        checkpointer=params['checkpointer']
    ).fit(params['epoch_counter'], train_loader, test_loader)