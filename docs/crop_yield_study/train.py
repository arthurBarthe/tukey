"""
Carries out training on the crop yield data
"""

import argparse
import warnings

import numpy as np
import torch
from ignite.handlers import EarlyStopping
from optuna.artifacts import upload_artifact, FileSystemArtifactStore
from sklearn.externals.array_api_compat import to_device
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tukey import losses
import optuna
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Loss
from tukey.models import NN

from .data_process import build_features_targets
from .data_process import dataset_from_arrays as dataset_from_arrays

parser = argparse.ArgumentParser()
parser.add_argument('--distribution', type=str, default="tukey")
parser.add_argument('--crop', type=str, default='maize')
args = parser.parse_args()

prob_dist = args.distribution
crop = args.crop

if prob_dist == 'gaussian':
    loss = losses.GaussianLoss()
elif prob_dist == "tukey":
    loss = losses.TuckeyGandHloss()


optuna_artifact_store = FileSystemArtifactStore('./crop_yield_study/artifacts')


years = np.arange(1982, 2016)
years_train = years[(np.mod(years, 10) != 5) & (np.mod(years, 10) != 6)]
years_validation = years[np.mod(years, 10) == 5]
years_test = years[np.mod(years, 10) == 6]

dataset_train, transform = dataset_from_arrays(*build_features_targets(years_train, crop=crop))
dataset_validation, _ = dataset_from_arrays(*build_features_targets(years_validation, crop=crop), transform=transform)
dataset_test, _ = dataset_from_arrays(*build_features_targets(years_test, crop=crop), transform=transform)


# create optuna study
study = optuna.create_study(study_name=f'crop_yield_study_{crop}_{prob_dist}_test',
                            storage='sqlite:///crop_yield_study/studies.db',
                            load_if_exists=True,
                            direction='minimize')

def objective(trial: optuna.Trial) -> float:
    try:
        # training parameters
        batch_size = 2 ** trial.suggest_int("batch_size", 8, 13)
        N_EPOCHS = 100

        # define loaders
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_validation = DataLoader(dataset_validation, batch_size=1024, shuffle=False)

        # Definition of the neural network
        size = 2 ** trial.suggest_int("layer_size", 4, 10)
        n_layers = trial.suggest_int("n_layers", 4, 8)
        nn_shape = (3,) + (size,) * n_layers
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        res_connections = trial.suggest_categorical("res_connections", [True, False])
        nn = NN(nn_shape, loss.n_required_channels, batch_norm=batch_norm, res=res_connections)  # 4 outputs required for G-and-H
        nn = nn.to(device='cuda:0')
        nn_parameters = nn.parameters()

        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        scheduler_step_size = trial.suggest_int("scheduler_step_size", 5, 20)
        optimizer = Adam(nn_parameters, lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size)

        # create trainer
        trainer = create_supervised_trainer(nn, optimizer, loss, device='cuda:0')
        val_metrics = {
            "loss": Loss(loss),
        }
        if prob_dist == "tukey":
            gmax = trial.suggest_uniform("gmax", 0.5, 2)
            hmax = trial.suggest_uniform("hmax", 0.2, 0.7)
            loss.gmax = gmax
            loss.hmax = hmax

        train_evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device='cuda:0')
        validation_evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device='cuda:0')

        early_stopper = EarlyStopping(patience=3, score_function=lambda engine: -engine.state.metrics["loss"],
                                      trainer=trainer, min_delta=0.01)
        validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

        @trainer.on(Events.ITERATION_COMPLETED(every=50))
        def log_training_loss(engine):
            print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_train_validation_evaluators(engine):
            train_evaluator.run(dataloader_train)
            validation_evaluator.run(dataloader_validation)
            train_metrics = train_evaluator.state.metrics
            validation_metrics = validation_evaluator.state.metrics
            print(f"Training Results - Epoch[{trainer.state.epoch}] Train loss: {train_metrics['loss']:.2f}  Validation loss: {validation_metrics['loss']:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_scheduler_step(engine):
            scheduler.step()

        trainer.run(dataloader_train, max_epochs=N_EPOCHS)

        # save trained model
        torch.save(nn, 'model.pth')
        artifact_id = upload_artifact(artifact_store=optuna_artifact_store,
                                      file_path='model.pth',
                                      study_or_trial=trial)
        trial.set_user_attr('model_artifact_id', artifact_id)
        import os
        os.remove('model.pth')

        return validation_evaluator.state.metrics["loss"]
    except ValueError as e:
        warnings.warn("Trial failed - returning infinite loss.")
        return np.inf

study.optimize(objective, n_trials=100)