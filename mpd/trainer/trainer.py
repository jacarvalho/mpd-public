import copy
from math import ceil

import numpy as np
import os
import time
import torch
import wandb
from collections import defaultdict
from tqdm.autonotebook import tqdm

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import dict_to_device, DEFAULT_TENSOR_ARGS, to_numpy


def get_num_epochs(num_train_steps, batch_size, dataset_len):
    return ceil(num_train_steps * batch_size / dataset_len)


def save_models_to_disk(models_prefix_l, epoch, total_steps, checkpoints_dir=None):
    for model, prefix in models_prefix_l:
        if model is not None:
            save_model_to_disk(model, epoch, total_steps, checkpoints_dir, prefix=f'{prefix}_')
            for submodule_key, submodule_value in model.submodules.items():
                save_model_to_disk(submodule_value, epoch, total_steps, checkpoints_dir,
                                   prefix=f'{prefix}_{submodule_key}_')


def save_model_to_disk(model, epoch, total_steps, checkpoints_dir=None, prefix='model_'):
    # If the model is frozen we do not save it again, since the parameters did not change
    if hasattr(model, 'is_frozen') and model.is_frozen:
        return

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}current_state_dict.pth'))
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}current.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}.pth'))


def save_losses_to_disk(train_losses, val_losses, checkpoints_dir=None):
    np.save(os.path.join(checkpoints_dir, f'train_losses.npy'), train_losses)
    np.save(os.path.join(checkpoints_dir, f'val_losses.npy'), val_losses)


class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # use -1 to deactivate it
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if self.patience == -1:
            return
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EMA:
    """
    https://github.com/jannerm/diffuser
    (empirical) exponential moving average parameters
    """

    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def do_summary(
        summary_fn,
        train_steps_current,
        model,
        batch_dict,
        loss_info,
        datasubset,
        **kwargs
):
    if summary_fn is None:
        return

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        summary_fn(train_steps_current,
                   model,
                   batch_dict=batch_dict,
                   loss_info=loss_info,
                   datasubset=datasubset,
                   **kwargs
                   )

    # set model to training mode
    model.train()


def train(model=None, train_dataloader=None, epochs=None, lr=None, steps_til_summary=None, model_dir=None, loss_fn=None,
          train_subset=None,
          summary_fn=None, steps_til_checkpoint=None,
          val_dataloader=None, val_subset=None,
          clip_grad=False,
          clip_grad_max_norm=1.0,
          val_loss_fn=None,
          optimizers=None, steps_per_validation=10, max_steps=None,
          use_ema: bool = True,
          ema_decay: float = 0.995, step_start_ema: int = 1000, update_ema_every: int = 10,
          use_amp=False,
          early_stopper_patience=-1,
          debug=False,
          tensor_args=DEFAULT_TENSOR_ARGS,
          **kwargs
          ):

    print(f'\n------- TRAINING STARTED -------\n')

    ema_model = None
    if use_ema:
        # Exponential moving average model
        ema = EMA(beta=ema_decay)
        ema_model = copy.deepcopy(model)

    # Model optimizers
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    # Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ## Build saving directories
    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Early stopping
    early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=0)

    stop_training = False
    train_steps_current = 0

    # save models before training
    save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')], 0, 0, checkpoints_dir)

    with tqdm(total=len(train_dataloader) * epochs, mininterval=1 if debug else 60) as pbar:
        train_losses_l = []
        validation_losses_l = []
        for epoch in range(epochs):
            model.train()  # set model to training mode
            for step, train_batch_dict in enumerate(train_dataloader):
                ####################################################################################################
                # TRAINING LOSS
                ####################################################################################################
                with TimerCUDA() as t_training_loss:
                    train_batch_dict = dict_to_device(train_batch_dict, tensor_args['device'])

                    # Compute losses
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        train_losses, train_losses_info = loss_fn(model, train_batch_dict, train_subset.dataset)

                    train_loss_batch = 0.
                    train_losses_log = {}
                    for loss_name, loss in train_losses.items():
                        single_loss = loss.mean()
                        train_loss_batch += single_loss
                        train_losses_log[loss_name] = to_numpy(single_loss).item()

                ####################################################################################################
                # SUMMARY
                if train_steps_current % steps_til_summary == 0:
                    # TRAINING
                    print(f"\n-----------------------------------------")
                    print(f"train_steps_current: {train_steps_current}")
                    print(f"t_training_loss: {t_training_loss.elapsed:.4f} sec")
                    print(f"Total training loss {train_loss_batch:.4f}")
                    print(f"Training losses {train_losses}")

                    train_losses_l.append((train_steps_current, train_losses_log))

                    with TimerCUDA() as t_training_summary:
                        do_summary(
                            summary_fn,
                            train_steps_current,
                            ema_model if ema_model is not None else model,
                            train_batch_dict,
                            train_losses_info,
                            train_subset,
                            prefix='TRAINING ',
                            debug=debug,
                            tensor_args=tensor_args
                        )
                    print(f"t_training_summary: {t_training_summary.elapsed:.4f} sec")

                    ################################################################################################
                    # VALIDATION LOSS and SUMMARY
                    validation_losses_log = {}
                    if val_dataloader is not None:
                        with TimerCUDA() as t_validation_loss:
                            print("Running validation...")
                            val_losses = defaultdict(list)
                            total_val_loss = 0.
                            for step_val, batch_dict_val in enumerate(val_dataloader):
                                batch_dict_val = dict_to_device(batch_dict_val, tensor_args['device'])
                                val_loss, val_loss_info = loss_fn(
                                    model, batch_dict_val, val_subset.dataset, step=train_steps_current)
                                for name, value in val_loss.items():
                                    single_loss = to_numpy(value)
                                    val_losses[name].append(single_loss)
                                    total_val_loss += np.mean(single_loss).item()

                                if step_val == steps_per_validation:
                                    break

                            validation_losses = {}
                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss).item()
                                validation_losses[f'VALIDATION {loss_name}'] = single_loss
                            print("... finished validation.")

                        print(f"t_validation_loss: {t_validation_loss.elapsed:.4f} sec")
                        print(f"Validation losses {validation_losses}")

                        validation_losses_log = validation_losses
                        validation_losses_l.append((train_steps_current, validation_losses_log))

                        # The validation summary is done only on one batch of the validation data
                        with TimerCUDA() as t_validation_summary:
                            do_summary(
                                summary_fn,
                                train_steps_current,
                                ema_model if ema_model is not None else model,
                                batch_dict_val,
                                val_loss_info,
                                val_subset,
                                prefix='VALIDATION ',
                                debug=debug,
                                tensor_args=tensor_args
                            )
                        print(f"t_valididation_summary: {t_validation_summary.elapsed:.4f} sec")

                    wandb.log({**train_losses_log, **validation_losses_log}, step=train_steps_current)

                ####################################################################################################
                # Early stopping
                if early_stopper.early_stop(total_val_loss):
                    print(f'Early stopped training at {train_steps_current} steps.')
                    stop_training = True

                ####################################################################################################
                # OPTIMIZE TRAIN LOSS BATCH
                with TimerCUDA() as t_training_optimization:
                    for optim in optimizers:
                        optim.zero_grad()

                    scaler.scale(train_loss_batch).backward()

                    if clip_grad:
                        for optim in optimizers:
                            scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=clip_grad_max_norm if isinstance(clip_grad, bool) else clip_grad
                        )

                    for optim in optimizers:
                        scaler.step(optim)

                    scaler.update()

                    if ema_model is not None:
                        if train_steps_current % update_ema_every == 0:
                            # update ema
                            if train_steps_current < step_start_ema:
                                # reset parameters ema
                                ema_model.load_state_dict(model.state_dict())
                            ema.update_model_average(ema_model, model)

                if train_steps_current % steps_til_summary == 0:
                    print(f"t_training_optimization: {t_training_optimization.elapsed:.4f} sec")

                ####################################################################################################
                # SAVING
                ####################################################################################################
                pbar.update(1)
                train_steps_current += 1

                if (steps_til_checkpoint is not None) and (train_steps_current % steps_til_checkpoint == 0):
                    save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                                        epoch, train_steps_current, checkpoints_dir)
                    save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)

                if stop_training or (max_steps is not None and train_steps_current == max_steps):
                    break

            if max_steps is not None and train_steps_current == max_steps:
                break

        # Update ema model at the end of training
        if ema_model is not None:
            # update ema
            if train_steps_current < step_start_ema:
                # reset parameters ema
                ema_model.load_state_dict(model.state_dict())
            ema.update_model_average(ema_model, model)

        # Save model at end of training
        save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                            epoch, train_steps_current, checkpoints_dir)
        save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)

        print(f'\n------- TRAINING FINISHED -------')
