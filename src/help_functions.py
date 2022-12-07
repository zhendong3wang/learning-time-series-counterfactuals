import os
import csv
import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

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
                    "fold_id",
                    "method",
                    "classifier_accuracy",
                    "autoencoder_loss",
                    "best_lr",
                    "proximity",
                    "validity",
                    "lof_score",
                    "relative_proximity",
                    "compactness",
                    "pred_margin_weight",
                    "step_weight_type",
                    "threshold_tau",
                ]
            )

    def write_result(
        self,
        fold_id,
        method_name,
        acc,
        ae_loss,
        best_lr,
        evaluate_res,
        pred_margin_weight=1.0,
        step_weight_type="",
        threshold_tau=0.5,
    ):
        (
            proxi,
            valid,
            lof_score,
            relative_proximity,
            compactness,
        ) = evaluate_res

        with open(self.file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.dataset_name,
                    fold_id,
                    method_name,
                    acc,
                    ae_loss,
                    best_lr,
                    proxi,
                    valid,
                    lof_score,
                    relative_proximity,
                    compactness,
                    pred_margin_weight,
                    step_weight_type,
                    threshold_tau,
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


def remove_paddings(cf_samples, padding_size):
    if padding_size != 0:
        # use np.squeeze() to cut the last time-series dimension, for evaluation
        cf_samples = np.squeeze(cf_samples[:, :-padding_size, :])
    else:
        cf_samples = np.squeeze(cf_samples)
    return cf_samples


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


def fit_evaluation_models(n_neighbors_lof, n_neighbors_nn, training_data):
    # Fit the LOF model for novelty detection (novelty=True)
    lof_estimator = LocalOutlierFactor(
        n_neighbors=n_neighbors_lof,
        novelty=True,
        metric="euclidean",
    )
    lof_estimator.fit(training_data)

    # Fit an unsupervised 1NN with all the training samples from the desired class
    nn_model = NearestNeighbors(n_neighbors=n_neighbors_nn, metric="euclidean")
    nn_model.fit(training_data)
    return lof_estimator, nn_model


def evaluate(
    X_pred_neg,
    cf_samples,
    pred_labels,
    cf_labels,
    lof_estimator_pos,
    lof_estimator_neg,
    nn_estimator_pos,
    nn_estimator_neg,
):
    proxi = euclidean_distance(X_pred_neg, cf_samples)
    valid = validity_score(pred_labels, cf_labels)
    compact = compactness_score(X_pred_neg, cf_samples)

    # TODO: add LOF and RP score for debugging training?
    lof_score = calculate_lof(
        cf_samples, pred_labels, lof_estimator_pos, lof_estimator_neg
    )
    rp_score = relative_proximity(
        X_pred_neg, cf_samples, pred_labels, nn_estimator_pos, nn_estimator_neg
    )

    return proxi, valid, lof_score, rp_score, compact


def euclidean_distance(X, cf_samples, average=True):
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances


def validity_score(pred_labels, cf_labels):
    desired_labels = 1 - pred_labels  # for binary classification
    return accuracy_score(y_true=desired_labels, y_pred=cf_labels)


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples):
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.01)

    # return a positive compactness, instead of 1 - np.mean(..)
    return np.mean(c, axis=(1, 0))


# def sax_compactness(X, cf_samples, n_timesteps):
#     from wildboar.transform import symbolic_aggregate_approximation

#     X = symbolic_aggregate_approximation(X, window=window, n_bins=n_bins)
#     cf_samples = symbolic_aggregate_approximation(
#         cf_samples, window=window, n_bins=n_bins
#     )

#     # absolute tolerance atol=0.01, 0.001, OR 0.0001?
#     c = np.isclose(X, cf_samples, atol=0.01)

#     return np.mean(1 - np.sum(c, axis=1) / n_timesteps)


def calculate_lof(cf_samples, pred_labels, lof_estimator_pos, lof_estimator_neg):
    desired_labels = 1 - pred_labels  # for binary classification

    pos_idx, neg_idx = (
        np.where(desired_labels == 1)[0],  # pos_label = 1
        np.where(desired_labels == 0)[0],  # neg_label - 0
    )
    # check if the NumPy array is empty
    if pos_idx.any():
        y_pred_cf1 = lof_estimator_pos.predict(cf_samples[pos_idx])
        n_error_cf1 = y_pred_cf1[y_pred_cf1 == -1].size
    else:
        n_error_cf1 = 0

    if neg_idx.any():
        y_pred_cf2 = lof_estimator_neg.predict(cf_samples[neg_idx])
        n_error_cf2 = y_pred_cf2[y_pred_cf2 == -1].size
    else:
        n_error_cf2 = 0

    lof_score = (n_error_cf1 + n_error_cf2) / cf_samples.shape[0]
    return lof_score


def relative_proximity(
    X_inputs, cf_samples, pred_labels, nn_estimator_pos, nn_estimator_neg
):
    desired_labels = 1 - pred_labels  # for binary classification

    nn_distance_list = np.array([])
    proximity_list = np.array([])

    pos_idx, neg_idx = (
        np.where(desired_labels == 1)[0],  # pos_label = 1
        np.where(desired_labels == 0)[0],  # neg_label = 0
    )
    if pos_idx.any():
        nn_distances1, _ = nn_estimator_pos.kneighbors(
            np.squeeze(X_inputs[pos_idx]), return_distance=True
        )
        proximity1 = euclidean_distance(
            X_inputs[pos_idx], cf_samples[pos_idx], average=False
        )

        nn_distance_list = np.concatenate(
            (nn_distance_list, np.squeeze(nn_distances1)), axis=0
        )
        proximity_list = np.concatenate((proximity_list, proximity1), axis=0)

    if neg_idx.any():
        nn_distances2, _ = nn_estimator_neg.kneighbors(
            np.squeeze(X_inputs[neg_idx]), return_distance=True
        )
        proximity2 = euclidean_distance(
            X_inputs[neg_idx], cf_samples[neg_idx], average=False
        )

        nn_distance_list = np.concatenate(
            (nn_distance_list, np.squeeze(nn_distances2)), axis=0
        )
        proximity_list = np.concatenate((proximity_list, proximity2), axis=0)

    # TODO: paired proximity score for (X_pred_neg, cf_samples), if not average (?)
    # relative_proximity = proximity / nn_distances.mean()
    relative_proximity = proximity_list.mean() / nn_distance_list.mean()

    return relative_proximity


"""
counterfactual model needed
"""


def find_best_lr(
    classifier,
    X_samples,
    pred_labels,
    autoencoder=None,
    encoder=None,
    decoder=None,
    lr_list=[0.001, 0.0001],
    pred_margin_weight=1.0,
    step_weights=None,
    random_state=None,
    padding_size=0,
    target_prob=0.5,
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
                probability=target_prob,
                only_encoder=encoder,
                only_decoder=decoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )
        else:
            cf_model = ModifiedLatentCF(
                probability=target_prob,
                autoencoder=autoencoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )

        cf_model.fit(classifier)

        if encoder and decoder:
            cf_embeddings, losses, _ = cf_model.transform(X_samples, pred_labels)
            cf_samples = decoder.predict(cf_embeddings)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_embeddings)
            cf_pred_labels = np.argmax(z_pred, axis=1)
        else:
            cf_samples, losses, _ = cf_model.transform(X_samples, pred_labels)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_samples)
            cf_pred_labels = np.argmax(z_pred, axis=1)

        valid_frac = validity_score(pred_labels, cf_pred_labels)
        proxi_score = euclidean_distance(
            remove_paddings(X_samples, padding_size),
            remove_paddings(cf_samples, padding_size),
        )

        # uncomment for debugging
        print(f"lr={lr} finished. Validity: {valid_frac}, proximity: {proxi_score}.")

        # TODO: fix (padding) dimensions of `lof_estimator` and `nn_estimator` during training, for debugging
        # proxi_score, valid_frac, lof_score, rp_score, cost_mean, cost_std = evaluate(
        #     X_pred_neg=X_samples,
        #     cf_samples=cf_samples,
        #     z_pred=z_pred,
        #     n_timesteps=_,
        #     lof_estimator=lof_estimator,
        #     nn_estimator=nn_estimator,
        # )

        # if valid_frac >= best_valid_frac and proxi_score <= best_proxi_score:
        if valid_frac >= best_valid_frac:
            best_cf_model, best_cf_samples = cf_model, cf_samples
            best_losses, best_lr, best_valid_frac = losses, lr, valid_frac
            if encoder and decoder:
                best_cf_embeddings = cf_embeddings

    return best_lr, best_cf_model, best_cf_samples, best_cf_embeddings
