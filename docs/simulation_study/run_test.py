"""
This scripts evaluates a trained model on test data.
"""
import os
import torch
import optuna
from ignite.engine import create_supervised_evaluator
from optuna.artifacts import download_artifact, FileSystemArtifactStore
from torch.utils.data import TensorDataset, DataLoader

optuna_artifact_store = FileSystemArtifactStore('./simulation_study/artifacts')

# define pinball loss
def pinball(yhat, y, alpha, loss):
    """compute pinball loss"""
    # convert yhat into predicted quantile
    qhat = loss.icdf(yhat, alpha)
    abs_distance = torch.abs(y - qhat)
    out = torch.zeros_like(qhat)
    idx = y > qhat
    out[idx] = alpha * abs_distance[idx]
    out[~idx] = (1 - alpha) * abs_distance[~idx]
    return out.mean()

def run_test(study_name):
    # retrieve the study
    study = optuna.load_study(study_name=study_name, storage="sqlite:///simulation_study/sim_study.db")
    dataset_id, prob_dist = study_name.split("_")
    dataset_id = int(dataset_id)

    # retrieve the model for the best trial
    best_trial = study.best_trial
    download_artifact(artifact_store=optuna_artifact_store,
                      file_path='model.pth',
                      artifact_id=best_trial.user_attrs['model_artifact_id'])
    model = torch.load('model.pth', weights_only=False)
    model = model.to(device='cuda:0')
    os.remove('model.pth')

    # retrieve the data
    from docs.simulation_study.generate import SimulationDataset
    dataset = SimulationDataset.load(dataset_id)
    x_data, y_data = dataset.sample()
    n_samples = len(x_data)

    TRAIN_SPLIT = int(n_samples * 0.8)
    TEST_SPLIT = int((n_samples - TRAIN_SPLIT) * 0.75)

    x_data, x_data_val = x_data[:TRAIN_SPLIT], x_data[TRAIN_SPLIT:]
    y_data, y_data_val = y_data[:TRAIN_SPLIT], y_data[TRAIN_SPLIT:]

    x_data_val, x_data_test = x_data_val[:TEST_SPLIT], x_data_val[TEST_SPLIT:]
    y_data_val, y_data_test = y_data_val[:TEST_SPLIT], y_data_val[TEST_SPLIT:]

    x_test_torch, y_test_torch = torch.tensor(x_data_test).reshape((-1, 1)), torch.tensor(y_data_test).reshape((-1, 1))

    dataset_test = TensorDataset(x_test_torch, y_test_torch)

    # run model on test dataset
    from ignite.metrics import Loss
    from tukey import losses

    if prob_dist == "tukey":
        loss = losses.TuckeyGandHloss(n_target_channels=1, hmax=2)
    elif prob_dist == "gaussian":
        loss = losses.GaussianLoss(n_target_channels=1)

    # we create the evaluator
    test_metrics = {
        "loss": Loss(loss),
        "q01": Loss(lambda x, y: pinball(x, y, 0.01, loss)),
        "q05": Loss(lambda x, y: pinball(x, y, 0.05, loss)),
        "q25": Loss(lambda x, y: pinball(x, y, 0.25, loss)),
        "q50": Loss(lambda x, y: pinball(x, y, 0.5, loss)),
        "q75": Loss(lambda x, y: pinball(x, y, 0.75, loss)),
        "q95": Loss(lambda x, y: pinball(x, y, 0.95, loss)),
        "q99": Loss(lambda x, y: pinball(x, y, 0.99, loss)),
    }

    test_loader = DataLoader(dataset_test, batch_size=1024)

    test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device='cuda:0')
    test_evaluator.run(test_loader)
    print(test_evaluator.state.metrics)
    # we store test metrics for the best trial as part of the study's attributes
    for metric_name, metric_value in test_evaluator.state.metrics.items():
        study.set_user_attr(metric_name, metric_value)


study_names = optuna.get_all_study_names(storage="sqlite:///simulation_study/sim_study.db")
for study_name in study_names:
    print(study_name)
    run_test(study_name)