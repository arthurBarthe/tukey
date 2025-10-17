"""
Script where we use the trained models on test data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from ignite.engine import create_supervised_evaluator, Events
from ignite.metrics import Loss
from mpmath import linspace
from optuna.artifacts import FileSystemArtifactStore, download_artifact
import torch
from torch.utils.data import DataLoader
from scipy.stats import probplot, uniform, norm

from docs.crop_yield_study.data_process import dataset_from_arrays as dataset_from_arrays, build_features_targets, CropYieldData
from tukey import losses
from tukey.utils import compute_pdf_tukey

from pathlib import Path
PAPER_PATH = Path("/home/user/Documents/6812476f991a0e1dd1cb719b")


def inv_transform(y):
    mu, sigma = transform[1]
    return y * sigma + mu

def store(x, y, y_pred):
    mean = loss.predict_mean(y_pred)
    res = loss.residuals(y_pred, y)
    thetas.append(torch.cat(loss.predict(y_pred), dim=-1))
    ys.append(y)
    pred_mean.append(inv_transform(mean.cpu()).numpy())
    residuals.append(res.cpu().numpy())
    return y_pred, y


crop = 'maize'
prob_dist = "tukey"

maizetukey = (crop == 'maize') and (prob_dist == 'tukey')
maizegaussian = (crop == 'maize') and (prob_dist == 'gaussian')

if prob_dist == 'gaussian':
    loss = losses.GaussianLoss()
elif prob_dist == "tukey":
    loss = losses.TuckeyGandHloss(n_target_channels=1)

optuna_artifact_store = FileSystemArtifactStore('./crop_yield_study/artifacts')

# download trained model from best trial
study = optuna.load_study(study_name=f'crop_yield_study_{crop}_{prob_dist}_test',
                          storage='sqlite:///crop_yield_study/studies.db')
best_trial = study.best_trial
download_artifact(artifact_store=optuna_artifact_store,
                  file_path="model.pth",
                  artifact_id=best_trial.user_attrs["model_artifact_id"])
model = torch.load('model.pth', weights_only=False)
os.remove('model.pth')

if prob_dist == 'tukey':
    loss.gmax = best_trial.params['gmax']
    loss.hmax = best_trial.params['hmax']

# obtain test dataset
years = np.arange(1982, 2016)
years_train = years[(np.mod(years, 10) != 5) & (np.mod(years, 10) != 6)]
years_validation = years[np.mod(years, 10) == 5]
years_test = years[np.mod(years, 10) == 6]

dataset_train, transform = dataset_from_arrays(*build_features_targets(years_train, crop=crop))
dataset_test, _ = dataset_from_arrays(*build_features_targets(years_test, crop=crop), transform=transform)

def pinball(yhat, y, alpha, loss):
    """compute pinball loss"""
    # convert yhat into predicted quantile
    qhat = loss.icdf(yhat, alpha)
    abs_distance = torch.abs(y - qhat)
    out = torch.zeros_like(qhat)
    idx = y > qhat
    out[idx] = alpha * abs_distance[idx]
    out[~idx] = (1 - alpha) * abs_distance[~idx]
    #
    out[idx] = 1 / (1 - alpha)
    out[~idx] = - 1 / alpha
    #
    out[idx] = 0
    out[~idx] = 1
    return out.mean()

def ci_covergage(yhat, y, p=0.5):
    """compute ci coverage"""
    lb, ub = loss.predict_ci(yhat, p)
    y_in_ci = (lb <= y) & (y <= ub)
    y_in_ci = y_in_ci * 1.
    return y_in_ci.mean()

def ci_len(yhat, y, p=0.5):
    """compute ci coverage"""
    lb, ub = loss.predict_ci(yhat, p)
    return (ub - lb).mean()


# create evaluator
test_metrics = {
    "loss": Loss(loss),
    "coverage50": Loss(ci_covergage),
    "len50": Loss(ci_len),
    "coverage95": Loss(lambda yhat, y: ci_covergage(yhat, y, 0.95)),
    "len95": Loss(lambda yhat, y: ci_len(yhat, y, 0.95)),
    "q01": Loss(lambda x, y: pinball(x, y, 0.01, loss)),
    "q05": Loss(lambda x, y: pinball(x, y, 0.05, loss)),
    "q25": Loss(lambda x, y: pinball(x, y, 0.25, loss)),
    "q50": Loss(lambda x, y: pinball(x, y, 0.5, loss)),
    "q75": Loss(lambda x, y: pinball(x, y, 0.75, loss)),
    "q95": Loss(lambda x, y: pinball(x, y, 0.95, loss)),
    "q99": Loss(lambda x, y: pinball(x, y, 0.99, loss)),
}

pred_mean, residuals, thetas, ys = [], [], [], []
test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device='cuda:0', output_transform=store)
test_evaluator.run(DataLoader(dataset_test, batch_size=1024, shuffle=False))
print(test_evaluator.state.metrics)

# qq plot residuals
residuals = np.concatenate(residuals).flatten()

fig = plt.figure()
ax = fig.add_subplot()
probplot(residuals, dist=norm(0, 1), fit=False, plot=ax)
ax.plot([-4, 4], [-4, 4], 'k--')
plt.show()



# PIT plot
residuals = norm.cdf(residuals)
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(residuals, bins=linspace(0, 1, 20), density=True, rwidth=0.98)
ax.plot([0, 1], [1, 1], 'k--')
ax.set_ylim(0, 2)
plt.show()

if maizetukey:
    fig.savefig(PAPER_PATH / 'res_val_qq_plot_maize.jpg', dpi=300)

if maizegaussian:
    fig.savefig(PAPER_PATH / 'res_val_qq_plot_maize_gaussian.jpg', dpi=300)


def plot_pdf(index, ys, thetas):
    thetas = torch.cat(thetas)
    ys = torch.cat(ys)
    y = ys[index: index + 1, :].cpu()
    theta = thetas[index: index + 1, :].cpu().repeat(1000, 1)
    zs = torch.linspace(-5, 15, 1000).reshape((-1, 1))
    pdfs = torch.exp(- loss.pointwise_likelihood(theta, zs))
    plt.figure()
    plt.plot(inv_transform(zs).flatten().numpy(), pdfs.flatten().numpy())
    plt.scatter([inv_transform(y), ], [0.01])
    plt.show()

metrics_results = [test_evaluator.state.metrics, ]
# year by year analysis
for year in years_test:
    pred_mean, residuals, thetas, ys = [], [], [], []
    test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device='cuda:0',
                                                 output_transform=store)
    dataset_test, _ = dataset_from_arrays(*build_features_targets([year, ], crop='maize'), transform=transform)
    loader = DataLoader(dataset_test, batch_size=1024, shuffle=False)
    test_evaluator.run(loader)
    metrics_results.append(test_evaluator.state.metrics)
    print(test_evaluator.state.metrics)

    pred_mean = np.concatenate(pred_mean).flatten()
    residuals = np.concatenate(residuals).flatten()
    residuals = norm.cdf(residuals)

    cydata = CropYieldData('maize', year)
    cydata.plot(cydata.flat_to_grid(pred_mean), cmap="cividis", vmin=0, vmax=10)
    plt.show()
    fig = cydata.plot(cydata.flat_to_grid(residuals), vmin=0, vmax=1, cmap='bwr')
    plt.show()

    if year == 2006 and maizetukey:
        fig.savefig(PAPER_PATH / 'residual__val_map_2.jpg', dpi=300)

    if year == 2006 and maizegaussian:
        fig.savefig(PAPER_PATH / 'residual__val_map_2_gaussian.jpg', dpi=300)

    continue
    # histogram of residuals
    plt.figure()
    plt.hist(residuals, bins=np.linspace(0, 1, 20))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    probplot(residuals, dist=norm(0, 1), fit=False, plot=ax)
    ax.plot([-4, 4], [-4, 4], 'k--')
    plt.show()


    # plot one prediction
    plot_pdf(0, ys, thetas)


# other plots
cydata = CropYieldData('maize', 2010)
fig = cydata.plot(vmin=0, vmax=15, cmap='cividis')
fig.savefig(PAPER_PATH / 'fig_example.jpg', dpi=300)


# make table
df_metrics = pd.DataFrame(metrics_results, index=["-", ] + [str(y) for y in years_test])
df_metrics.to_csv(f"crop_yield_study/test_metrics_{crop}_{prob_dist}.csv", index_label="year")

try:
    df_metrics_gaussian = pd.read_csv(f"crop_yield_study/test_metrics_{crop}_gaussian.csv")
    df_metrics_tukey = pd.read_csv(f"crop_yield_study/test_metrics_{crop}_tukey.csv")
    df_metrics_gaussian["method"] = "gaussian"
    df_metrics_tukey["method"] = "tukey"
    df_metrics_gaussian = df_metrics_gaussian.set_index("method", append=True)
    df_metrics_tukey = df_metrics_tukey.set_index("method", append=True)
    result_df = pd.concat([df_metrics_gaussian, df_metrics_tukey]).sort_index()
    result_df.index = result_df.index.set_names('', level=0)
    print(result_df)
    print("***********")
    print(result_df.to_latex(float_format="%.2f"))
    print("***********")
except FileNotFoundError:
    pass


# push figures to overleaf
import subprocess

command = f"cd {PAPER_PATH} && git pull"
result = subprocess.run(command, shell=True, capture_output=True)

# Command to change directory and run git push
command = f"cd {PAPER_PATH} && git commit -am \"automatic figure update\""
result = subprocess.run(command, shell=True, capture_output=True)

# Print the output
print("Output:", result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

command = f"cd {PAPER_PATH} && git push"
result = subprocess.run(command, shell=True, capture_output=True)

# Print the output
print("Output:", result.stdout)
if result.stderr:
    print("Errors:", result.stderr)