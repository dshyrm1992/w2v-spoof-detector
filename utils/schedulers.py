import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class LinearSchedulerWithWarmup:
    def __init__(self, initial_value, final_value, warmup_steps, constant_steps, linear_steps):
        self.losses = []
        self.n_steps = 0
        self.value_at_epoch = torch.cat((
            torch.linspace(0, initial_value, steps=warmup_steps),
            torch.full((constant_steps,), initial_value),
            torch.linspace(initial_value, final_value, linear_steps),
        ))

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.value_at_epoch[self.n_steps] if self.n_steps < len(self.value_at_epoch) else self.value_at_epoch[-1]

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]
