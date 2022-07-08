import os
import csv
import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample, shuffle

# from _composite import ModifiedLatentCF
from _guided import ModifiedLatentCF
from _vanilla import LatentCF

# from keras import backend as K


class ResultWriter:
    def __init__(self, file_name, dataset_name):
        self.file_name = file_name
        self.dataset_name = dataset_name

    def write_head(self):
        # write the head in csv file
        with open(self.file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "method",
                    "classifier_accuracy",
                    "autoencoder_loss",
                    "best_lr",
                    "proximity",
                    "validity",
                    "margin_mean",
                    "margin_std",
                    "pred_margin_weight",
                    "step_weight_type",
                ]
            )

    def write_result(
        self,
        method_name,
        acc,
        ae_loss,
        best_lr,
        evaluate_res,
        pred_margin_weight=1.0,
        step_weight_type="",
    ):
        proxi, valid, cost_mean, cost_std = evaluate_res

        with open(self.file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.dataset_name,
                    method_name,
                    acc,
                    ae_loss,
                    best_lr,
                    proxi,
                    valid,
                    cost_mean,
                    cost_std,
                    pred_margin_weight,
                    step_weight_type,
                ]
            )


"""
time series scaling
"""


def time_series_normalize(data, n_timesteps, n_features=1, scaler=None):
    # reshape data to 1 column
    data_reshaped = data.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_reshaped)

    normalized = scaler.transform(data_reshaped)

    # return reshaped data into [samples, timesteps, features]
    return normalized.reshape(-1, n_timesteps, n_features), scaler


def time_series_revert(normalized_data, n_timesteps, n_features=1, scaler=None):
    # reshape data to 1 column
    data_reshaped = normalized_data.reshape(-1, 1)

    reverted = scaler.inverse_transform(data_reshaped)

    # return reverted data into [samples, timesteps, features]
    return reverted.reshape(-1, n_timesteps, n_features)


"""
data pre-processing
"""


def conditional_pad(X):
    num = X.shape[1]

    if num % 4 != 0:
        # find the next integer that can be divided by 4
        next_num = (int(num / 4) + 1) * 4
        padding_size = next_num - num
        X_padded = np.pad(
            X, pad_width=((0, 0), (0, padding_size), (0, 0))
        )  # pad for 3d array

        return X_padded, padding_size

    # else return the original X
    return X, 0  # padding size = 0


# Upsampling the minority class


def upsample_minority(X, y, pos_label=1, neg_label=0, random_state=39):
    # Get counts
    pos_counts = pd.value_counts(y)[pos_label]
    neg_counts = pd.value_counts(y)[neg_label]
    # Divide by class
    X_pos, X_neg = X[y == pos_label], X[y == neg_label]

    if pos_counts == neg_counts:
        # Balanced dataset
        return X, y
    elif pos_counts > neg_counts:
        # Imbalanced dataset
        X_neg_over = resample(
            X_neg, replace=True, n_samples=pos_counts, random_state=random_state
        )
        X_concat = np.concatenate([X_pos, X_neg_over], axis=0)
        y_concat = np.array(
            [pos_label for i in range(pos_counts)]
            + [neg_label for j in range(pos_counts)]
        )
    else:
        # Imbalanced dataset
        X_pos_over = resample(
            X_pos, replace=True, n_samples=neg_counts, random_state=random_state
        )
        X_concat = np.concatenate([X_pos_over, X_neg], axis=0)
        y_concat = np.array(
            [pos_label for i in range(neg_counts)]
            + [neg_label for j in range(neg_counts)]
        )

    # Shuffle the index after up-sampling
    X_concat, y_concat = shuffle(X_concat, y_concat, random_state=random_state)

    return X_concat, y_concat


"""
deep models needed
"""
# Method: For plotting the accuracy/loss of keras models


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


# Method: Fix the random seeds to get consistent models


def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)


"""
evaluation metrics
"""


def evaluate(X_pred_neg, best_cf_samples, z_pred, n_timesteps):
    proxi = euclidean_distance(X_pred_neg, best_cf_samples)
    valid = validity_score(z_pred)
    # compact = compactness_score(X_pred_neg, best_cf_samples, n_timesteps=n_timesteps)
    cost_mean, cost_std = cost_score(z_pred)

    return proxi, valid, cost_mean, cost_std


def euclidean_distance(X, cf_samples):
    distance = np.mean(np.linalg.norm(X - cf_samples, axis=1))
    return distance


def validity_score(cf_probs, decision_prob=0.5):
    valid_counts = np.sum(cf_probs >= decision_prob)
    total_counts = len(cf_probs)
    return valid_counts / total_counts


def cost_score(cf_probs, decision_prob=0.5):
    diff = cf_probs - decision_prob
    return np.mean(diff), np.std(diff)


"""
counterfactual model needed
"""


def find_best_alpha(autoencoder, classifier, X_samples, alpha_list=[0.001, 0.0001]):
    # Find the best alpha for vanilla LatentCF
    best_cf_model, best_cf_samples = None, None
    best_losses, best_valid_frac, best_alpha = 0, -1, 0

    for alp in alpha_list:
        # Fit the LatentCF model
        cf_model = LatentCF(probability=0.5, alpha=alp, autoencoder=autoencoder)
        cf_model.fit(classifier)

        cf_samples, losses = cf_model.transform(X_samples)
        # predicted probabilities of CFs
        z_pred = classifier.predict(cf_samples)

        print(f"alpha={alp} finished.")
        valid_frac = validity_score(z_pred)

        if valid_frac >= best_valid_frac:
            best_cf_model, best_cf_samples = cf_model, cf_samples
            best_losses, best_alpha, best_valid_frac = losses, alp, valid_frac

    return best_alpha, best_cf_model, best_cf_samples


def find_best_lr(
    classifier,
    X_samples,
    autoencoder=None,
    encoder=None,
    decoder=None,
    lr_list=[0.001, 0.0001],
    pred_margin_weight=1.0,
    step_weights=None,
    random_state=None,
):
    # Find the best alpha for vanilla LatentCF
    best_cf_model, best_cf_samples, best_cf_embeddings = None, None, None
    best_losses, best_valid_frac, best_lr = 0, -1, 0

    for lr in lr_list:
        print(f"======================== CF search started, with lr={lr}.")
        # Fit the LatentCF model
        # TODO: fix the class name here: ModifiedLatentCF or GuidedLatentCF? from _guided or _composite?
        if encoder and decoder:
            cf_model = ModifiedLatentCF(
                probability=0.5,
                only_encoder=encoder,
                only_decoder=decoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )
        else:
            cf_model = ModifiedLatentCF(
                probability=0.5,
                autoencoder=autoencoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )

        cf_model.fit(classifier)

        if encoder and decoder:
            cf_embeddings, losses, _ = cf_model.transform(X_samples)
            cf_samples = decoder.predict(cf_embeddings)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_embeddings)[:, 1]
        else:
            cf_samples, losses, _ = cf_model.transform(X_samples)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_samples)[:, 1]

        valid_frac = validity_score(z_pred)
        proxi_score = euclidean_distance(X_samples, cf_samples)
        proxi_score, valid_frac, cost_mean, cost_std = evaluate(
            X_samples, cf_samples, z_pred, _
        )
        # uncomment for debugging
        print(
            f"lr={lr} finished. Validity: {valid_frac}, proximity (with padding): {proxi_score}, margin difference: {cost_mean,cost_std}."
        )

        # if valid_frac >= best_valid_frac and proxi_score <= best_proxi_score:
        if valid_frac >= best_valid_frac:
            best_cf_model, best_cf_samples = cf_model, cf_samples
            best_losses, best_lr, best_valid_frac = losses, lr, valid_frac
            if encoder and decoder:
                best_cf_embeddings = cf_embeddings

    return best_lr, best_cf_model, best_cf_samples, best_cf_embeddings
