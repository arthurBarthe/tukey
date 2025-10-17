import numpy as np
import matplotlib.pyplot as plt
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss
from scipy.stats import t

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from optuna.artifacts import upload_artifact, download_artifact, FileSystemArtifactStore

import argparse
from tukey import losses
from tukey.transforms import SoftPlusTransform
from tukey.models import NN

import optuna

#############
optuna_artifact_store = FileSystemArtifactStore('./simulation_study/artifacts')


parser = argparse.ArgumentParser()
parser.add_argument('--datasetId', type=int, default=-1)

args = parser.parse_args()

# data parameters
DATASET_ID = args.datasetId

# obtain the data
from docs.simulation_study.generate import SimulationDataset
dataset = SimulationDataset.load(DATASET_ID)
x_data, y_data = dataset.sample()
n_samples = len(x_data)

TRAIN_SPLIT = int(n_samples * 0.8)
TEST_SPLIT = int((n_samples - TRAIN_SPLIT) * 0.75)
MISSPECIFIED = False

PROB_DIST = "gaussian"

x_data, x_data_val = x_data[:TRAIN_SPLIT], x_data[TRAIN_SPLIT:]
y_data, y_data_val = y_data[:TRAIN_SPLIT], y_data[TRAIN_SPLIT:]

x_data_val, x_data_test = x_data_val[:TEST_SPLIT], x_data_val[TEST_SPLIT:]
y_data_val, y_data_test = y_data_val[:TEST_SPLIT], y_data_val[TEST_SPLIT:]

x_torch, y_torch = torch.tensor(x_data).reshape((-1, 1)), torch.tensor(y_data).reshape((-1, 1))
x_val_torch, y_val_torch = torch.tensor(x_data_val).reshape((-1, 1)), torch.tensor(y_data_val).reshape((-1, 1))
x_test_torch, y_test_torch = torch.tensor(x_data_test).reshape((-1, 1)), torch.tensor(y_data_test).reshape((-1, 1))


dataset = TensorDataset(x_torch, y_torch)
dataset_val = TensorDataset(x_val_torch, y_val_torch)
dataset_test = TensorDataset(x_test_torch, y_test_torch)

# optuna study to optimize hyperparameters
study = optuna.create_study(study_name=f"{DATASET_ID}_{PROB_DIST}",
                            direction='minimize',
                            storage="sqlite:///simulation_study/sim_study.db",
                            load_if_exists=True)
study.set_user_attr("dataset_id", DATASET_ID)
study.set_user_attr("prob_dist", PROB_DIST)

if PROB_DIST == "tukey":
    loss = losses.TuckeyGandHloss(n_target_channels=1, hmax=2)
elif PROB_DIST == "gaussian":
    loss = losses.GaussianLoss(n_target_channels=1)

def objective(trial: optuna.Trial) -> float:
    try:
        tensorboard_writer = SummaryWriter(f"simulation_study/tensorboard/{study.study_name}/" + str(trial.number))

        # training parameters
        batch_size = 2 ** trial.suggest_int("batch_size", 8, 12)
        N_EPOCHS = 50

        # Definition of the neural network
        size = 2 ** trial.suggest_int("layer_size", 4, 10)
        n_layers = trial.suggest_int("n_layers", 2, 6)
        nn_shape = (1, ) + (size, ) * n_layers
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        nn = NN(nn_shape, loss.n_required_channels, batch_norm=batch_norm)  # 4 outputs required for G-and-H
        nn = nn.to(device='cuda:0')
        nn_parameters = nn.parameters()

        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        scheduler_step_size = trial.suggest_int("scheduler_step_size", 5, 20)
        optimizer = Adam(nn_parameters, lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size)

        # define trainers
        from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
        from ignite.metrics import Loss
        from ignite.handlers import ModelCheckpoint, EarlyStopping

        trainer = create_supervised_trainer(nn, optimizer, loss, device='cuda:0')

        # add a profiler
        basic_profiler = BasicTimeProfiler()
        basic_profiler.attach(trainer)

        val_metrics = {
            "loss": Loss(loss),
        }

        train_evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device='cuda:0')
        val_evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device='cuda:0')
        early_stopper = EarlyStopping(patience=3, score_function=lambda engine: -engine.state.metrics["loss"], trainer=trainer)

        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

        @trainer.on(Events.ITERATION_COMPLETED(every=5))
        def log_training_loss(engine):
            print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")


        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            train_evaluator.run(dataloader)
            val_evaluator.run(val_loader)
            train_metrics = train_evaluator.state.metrics
            val_metrics = val_evaluator.state.metrics
            print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {train_metrics['loss']:.2f}")
            trial.report(val_metrics["loss"], trainer.state.epoch)
            tensorboard_writer.add_scalar("training_loss", train_metrics["loss"], trainer.state.epoch)
            tensorboard_writer.add_scalar("validation_loss", val_metrics["loss"], trainer.state.epoch)
            scheduler.step()

        trainer.run(dataloader, max_epochs=N_EPOCHS)
        basic_profiler.write_results('profiler_results.csv')
        artifact_id = upload_artifact(artifact_store=optuna_artifact_store,
                                      file_path="profiler_results.csv",
                                      study_or_trial=trial)
        trial.set_user_attr("profiler_artifact_id", artifact_id)
        # save trained model
        torch.save(nn, 'model.pth')
        artifact_id = upload_artifact(artifact_store=optuna_artifact_store,
                                      file_path='model.pth',
                                      study_or_trial=trial)
        trial.set_user_attr('model_artifact_id', artifact_id)
        import os
        os.remove('model.pth')

        tensorboard_writer.flush()
        tensorboard_writer.close()
        return val_evaluator.state.metrics["loss"]
    except Exception as e:
        return np.inf


study.optimize(objective, n_trials=50)

# Upon completion of the study, we evaluate the best model on test data
# First we retrive the best trial and the model's parameters
best_trial = study.best_trial
download_artifact(artifact_store=optuna_artifact_store,
                  file_path='model.pth',
                  artifact_id=best_trial.user_attrs['model_artifact_id'])
model = torch.load('model.pth', weights_only=False)
model = model.to(device='cuda:0')

# we create the evaluator
test_metrics = {
    "loss": Loss(loss),
}

test_loader = DataLoader(dataset_test)

test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device='cuda:0')
test_evaluator.run(test_loader)
# we store test metrics for the best trial as part of the study's attributes
for metric_name, metric_value in test_evaluator.state.metrics.items():
    study.set_user_attr(metric_name, metric_value)