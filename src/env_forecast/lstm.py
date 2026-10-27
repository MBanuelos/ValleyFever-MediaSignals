"""Direct LSTM forecaster for the harness (one network per county, origin, horizon and feature set)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .tasks import ForecastTask


class _Network(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


def monthly_inputs(task: ForecastTask) -> pd.DataFrame:
    """
    One row per month k holding what was published by the end of k: rate[k - case lag] (R, RE),
    every environmental column x[k - its availability lag] (E, RE), and sin/cos of k's calendar month.

    A sequence ending at pseudo-origin s therefore only contains values published by s.
    """
    config, history = task.config, task.history
    columns = {}
    if task.feature_set in ("R", "RE"):
        columns["rate"] = history[config.target_col].shift(config.rate.availability_lag)
    if task.feature_set in ("E", "RE"):
        for col, lag in config.env_cutoffs.items():
            columns[col] = history[col].shift(lag)
    month = history.index.month.to_numpy()
    columns["month_sin"] = pd.Series(np.sin(2 * np.pi * month / 12), index=history.index)
    columns["month_cos"] = pd.Series(np.cos(2 * np.pi * month / 12), index=history.index)
    return pd.DataFrame(columns, index=history.index)


@dataclass
class LSTMModel:
    """
    Direct LSTM: sequences of `lookback` monthly input rows ending at each pseudo-origin s predict y[s + h].

    Preprocessing uses training data only: input and target standardisation come from the training
    sequences, and missing inputs become the training mean. Early stopping monitors the last
    `validation_fraction` of training sequences (chronologically); with refit_full the network is
    then retrained on all training sequences for the selected number of epochs. Runs single-threaded
    and seeded by task.seed, so results do not depend on parallelism.
    """

    lookback: int = 12
    hidden_size: int = 8
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 30
    validation_fraction: float = 0.2
    refit_full: bool = True
    log_target: bool = False
    name: str = "lstm"
    uses_features: bool = True

    def fit_predict(self, task: ForecastTask) -> float:
        inputs = monthly_inputs(task)
        position = {month: i for i, month in enumerate(inputs.index)}
        rows = [s for s in task.X_train.index if position[s] >= self.lookback - 1]
        if len(rows) < 10:
            return np.nan
        last_train_row = position[max(rows)]
        scale_rows = inputs.iloc[position[min(rows)] - self.lookback + 1:last_train_row + 1]
        mean, std = scale_rows.mean(), scale_rows.std().replace(0, 1).fillna(1)
        scaled = ((inputs - mean) / std).fillna(0.0).to_numpy(np.float32)

        def sequence(s: pd.Period) -> np.ndarray:
            end = position[s]
            return scaled[end - self.lookback + 1:end + 1]

        X = np.stack([sequence(s) for s in rows])
        y = task.y_train.loc[rows].to_numpy(float)
        y = np.log1p(y) if self.log_target else y
        y_mean, y_std = y.mean(), y.std() or 1.0
        y = ((y - y_mean) / y_std).astype(np.float32)

        torch.set_num_threads(1)
        n_validation = max(1, int(round(self.validation_fraction * len(rows))))
        network, epochs = self._train(X[:-n_validation], y[:-n_validation], X[-n_validation:], y[-n_validation:], task.seed)
        if self.refit_full:
            network, _ = self._train(X, y, None, None, task.seed, fixed_epochs=epochs)
        network.eval()
        with torch.no_grad():
            prediction = network(torch.from_numpy(sequence(task.origin)[None].copy())).item()
        prediction = prediction * y_std + y_mean
        prediction = np.expm1(prediction) if self.log_target else prediction
        return float(max(prediction, 0.0))

    def _train(self, X, y, X_val, y_val, seed: int, fixed_epochs: int | None = None):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        network = _Network(X.shape[2], self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
        best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0
        for epoch in range(1, (fixed_epochs or self.max_epochs) + 1):
            network.train()
            for batch in torch.randperm(len(X_t), generator=generator).split(self.batch_size):
                optimizer.zero_grad()
                loss_fn(network(X_t[batch]), y_t[batch]).backward()
                optimizer.step()
            if fixed_epochs is not None:
                continue
            network.eval()
            with torch.no_grad():
                val_loss = loss_fn(network(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item()
            if val_loss < best_loss:
                best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(network.state_dict()), epoch, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if fixed_epochs is None and best_state is not None:
            network.load_state_dict(best_state)
        return network, (fixed_epochs or best_epoch)
