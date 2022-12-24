#!/usr/bin/env python
# coding: utf-8
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from wildboar.datasets import load_dataset

from help_functions import (
    ResultWriter,
    conditional_pad,
    remove_paddings,
    evaluate,
    find_best_lr,
    reset_seeds,
    time_series_normalize,
    upsample_minority,
    fit_evaluation_models,
)
from keras_models import *
from _guided import get_global_weights

os.environ["TF_DETERMINISTIC_OPS"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def main():
    parser = ArgumentParser(description="Run this script to evaluate LatentCF method.")
    parser.add_argument(
        "--dataset", type=str, help="Dataset that the experiment is running on."
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=1,
        help="The positive label of the dataset, e.g. 1 or 2.",
    )
    parser.add_argument(
        "--neg",
        type=int,
        default=0,
        help="The negative label of the dataset, e.g. 0 or -1",
    )
    parser.add_argument("--output", type=str, help="Output file name.")
    parser.add_argument(
        "--n-lstmcells",
        type=int,
        default=8,
        help="Boolean flag of using (relatively) shallower CNN model, default False.",
    )
    parser.add_argument(
        "--lr-list",
        nargs="+",
        type=float,
        required=False,
        default=[None, None, None],
        help="Learning rates for each CF search model, if None then automatically search between [0.001, 0.0001]",
    )
    parser.add_argument(
        "--w-type",
        type=str,
        default="local",
        help="Local, global, uniform, or unconstrained.",
    )
    parser.add_argument(
        "--w-value",
        type=float,
        default=0.5,
        help="The weight value for prediction margin loss, ranging between [0, 1]. Equals to 1 refer to unconstrained version.",
    )
    parser.add_argument(
        "--tau-value",
        type=float,
        default=0.5,
        help="The threshold of decision boundary during CF search, ranging between [0.5, 1], default to 0.5.",
    )
    A = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    logger.info(f"LR list: {A.lr_list}.")  # for debugging
    logger.info(f"W type: {A.w_type}.")  # for debugging

    RANDOM_STATE = 39
    # PRED_MARGIN_W_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] # TODO: write another script for ablation study
    pred_margin_weight = A.w_value
    logger.info(f"W value: {pred_margin_weight}.")  # for debugging
    logger.info(f"Tau value: {A.tau_value}.")  # for debugging

    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)
    logger.info(f"Result writer is ready, writing to {A.output}...")
    # If `A.output` file already exists, no need to write head (directly append)
    if not os.path.isfile(A.output):
        result_writer.write_head()

    # 1. Load data
    X, y = load_dataset(A.dataset, repository="wildboar/ucr")

    # Convert positive and negative labels to 1 and 0
    pos_label, neg_label = 1, 0
    y_copy = y.copy()
    if A.pos != pos_label:
        y_copy[y == A.pos] = pos_label  # convert/normalize positive label to 1
    if A.neg != neg_label:
        y_copy[y == A.neg] = neg_label  # convert negative label to 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_idx = 0
    for train_index, test_index in skf.split(X, y_copy):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_copy[train_index], y_copy[test_index]

        # Get 50 samples for CF evaluation if test size larger than 50
        test_size = len(y_test)
        if test_size >= 50:
            try:
                test_indices = np.arange(test_size)
                _, _, _, rand_test_idx = train_test_split(
                    y_test,
                    test_indices,
                    test_size=50,
                    random_state=RANDOM_STATE,
                    stratify=y_test,
                )
            except ValueError:  # ValueError: The train_size = 1 should be greater or equal to the number of classes = 2
                rand_test_idx = np.arange(test_size)

        else:
            rand_test_idx = np.arange(test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        fold_idx += 1
        logger.info(
            f"Current CV fold: [{fold_idx}], with `X_train`, `X_val` & `X_test` shape: {X_train.shape} | {X_val.shape} | {X_test.shape}."
        )

        # Upsample the minority class
        y_train_copy = y_train.copy()
        X_train, y_train = upsample_minority(
            X_train, y_train, pos_label=pos_label, neg_label=neg_label
        )
        if y_train.shape != y_train_copy.shape:
            logger.info(
                f"Data upsampling performed, current distribution of y_train: \n{pd.value_counts(y_train)}."
            )
        else:
            logger.info(
                f"Current distribution of y_train: \n{pd.value_counts(y_train)}."
            )

        nb_classes = len(np.unique(y_train))
        y_train_classes, y_val_classes, y_test_classes = (
            y_train.copy(),
            y_val.copy(),
            y_test.copy(),
        )
        y_train, y_val, y_test = (
            to_categorical(y_train, nb_classes),
            to_categorical(y_val, nb_classes),
            to_categorical(y_test, nb_classes),
        )

        # ### 1.1 Normalization - fit scaler using training data
        n_training, n_timesteps = X_train.shape
        n_features = 1

        X_train_processed, trained_scaler = time_series_normalize(
            data=X_train, n_timesteps=n_timesteps
        )
        X_val_processed, _ = time_series_normalize(
            data=X_val, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        X_test_processed, _ = time_series_normalize(
            data=X_test, n_timesteps=n_timesteps, scaler=trained_scaler
        )

        # add extra padding zeros if n_timesteps cannot be divided by 4, required for 1dCNN autoencoder structure
        X_train_processed_padded, padding_size = conditional_pad(X_train_processed)
        X_val_processed_padded, _ = conditional_pad(X_val_processed)
        X_test_processed_padded, _ = conditional_pad(X_test_processed)
        n_timesteps_padded = X_train_processed_padded.shape[1]
        logger.info(
            f"Data pre-processed, original #timesteps={n_timesteps}, padded #timesteps={n_timesteps_padded}."
        )

        # ### 1.2 Evaluation models
        n_neighbors_lof = int(np.cbrt(X_train_processed.shape[0]))
        lof_estimator_pos, nn_model_pos = fit_evaluation_models(
            n_neighbors_lof=n_neighbors_lof,
            n_neighbors_nn=1,
            training_data=np.squeeze(X_train_processed[y_train_classes == pos_label]),
        )
        lof_estimator_neg, nn_model_neg = fit_evaluation_models(
            n_neighbors_lof=n_neighbors_lof,
            n_neighbors_nn=1,
            training_data=np.squeeze(X_train_processed[y_train_classes == neg_label]),
        )
        logger.info(
            f"LOF and NN estimators trained for dataset: [[{A.dataset}]], fold-ID: {fold_idx}."
        )

        # ## 2. LatentCF models
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds()

        ###############################################
        # ## 2.0 LSTM-FCN classifier
        ###############################################
        # ### LSTM-FCN classifier
        classifier = LSTMFCNClassifier(
            n_timesteps_padded, n_features, n_output=2, n_LSTM_cells=A.n_lstmcells
        )

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        classifier.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Define the early stopping criteria
        early_stopping_accuracy = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30, restore_best_weights=True
        )
        # Train the model
        reset_seeds()
        logger.info("Training log for LSTM-FCN classifier:")
        classifier_history = classifier.fit(
            X_train_processed_padded,
            y_train,
            epochs=150,
            batch_size=32,
            shuffle=True,
            verbose=True,
            validation_data=(X_val_processed_padded, y_val),
            callbacks=[early_stopping_accuracy],
        )

        y_pred = classifier.predict(X_test_processed_padded)
        y_pred_classes = np.argmax(y_pred, axis=1)
        acc = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes)
        logger.info(f"LSTM-FCN classifier trained, with test accuracy {acc}.")

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(
                y_true=y_test_classes, y_pred=y_pred_classes, labels=[1, 0]
            ),
            index=["True:pos", "True:neg"],
            columns=["Pred:pos", "Pred:neg"],
        )
        logger.info(f"Confusion matrix: \n{confusion_matrix_df}.")

        # ### 2.0.1 Get `step_weights` based on the input argument
        if A.w_type == "global":
            step_weights = get_global_weights(
                X_train_processed_padded,
                y_train_classes,
                classifier,
                random_state=RANDOM_STATE,
            )
        elif A.w_type == "uniform":
            step_weights = np.ones((1, n_timesteps_padded, n_features))
        elif A.w_type.lower() == "local":
            step_weights = "local"
        elif A.w_type == "unconstrained":
            step_weights = np.zeros((1, n_timesteps_padded, n_features))
        else:
            raise NotImplementedError(
                "A.w_type not implemented, please choose 'local', 'global', 'uniform', or 'unconstrained'."
            )

        ###############################################
        # ## 2.1 CF search with 1dCNN autoencoder
        ###############################################
        # ### 1dCNN autoencoder
        autoencoder = Autoencoder(n_timesteps_padded, n_features)
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        autoencoder.compile(optimizer=optimizer, loss="mse")

        # Define the early stopping criteria
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=5, restore_best_weights=True
        )
        # Train the model
        reset_seeds()
        logger.info("Training log for 1dCNN autoencoder:")
        autoencoder_history = autoencoder.fit(
            X_train_processed_padded,
            X_train_processed_padded,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=True,
            validation_data=(X_val_processed_padded, X_val_processed_padded),
            callbacks=[early_stopping],
        )

        ae_val_loss = np.min(autoencoder_history.history["val_loss"])
        logger.info(f"1dCNN autoencoder trained, with validation loss: {ae_val_loss}.")

        ### Evaluation metrics
        # update: only evaluating one prediction_margin_weight in this script
        logger.info(
            f"The current prediction margin weight is {pred_margin_weight}, for [1dCNN autoencoder]."
        )

        # Get these instances for CF evaluation; class abnormal (0) VS normal class (1)
        rand_X_test = X_test_processed_padded[rand_test_idx]
        rand_y_pred = y_pred_classes[rand_test_idx]

        lr_list = [A.lr_list[0]] if A.lr_list[0] is not None else [0.001, 0.0001]
        best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
            classifier,
            X_samples=rand_X_test,
            pred_labels=rand_y_pred,
            autoencoder=autoencoder,
            lr_list=lr_list,
            pred_margin_weight=pred_margin_weight,
            step_weights=step_weights,
            random_state=RANDOM_STATE,
            padding_size=padding_size,
            target_prob=A.tau_value,
        )
        logger.info(f"The best learning rate found is {best_lr}.")

        # predicted probabilities of CFs
        z_pred = classifier.predict(best_cf_samples)
        cf_pred_labels = np.argmax(z_pred, axis=1)

        # remove extra paddings after counterfactual generation in 1dCNN autoencoder
        best_cf_samples = remove_paddings(best_cf_samples, padding_size)
        # use the unpadded X_test for evaluation
        rand_X_test_original = np.squeeze(X_test_processed[rand_test_idx])

        evaluate_res = evaluate(
            rand_X_test_original,
            best_cf_samples,
            rand_y_pred,
            cf_pred_labels,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "1dCNN autoencoder",
            acc,
            ae_val_loss,
            best_lr,
            evaluate_res,
            pred_margin_weight=pred_margin_weight,
            step_weight_type=A.w_type.lower(),
            threshold_tau=A.tau_value,
        )
        logger.info(
            f"Done for CF search [1dCNN autoencoder], pred_margin_weight={pred_margin_weight}."
        )

        # ###############################################
        # ## 2.2 CF search with LSTM autoencoder
        ################################################
        # ### LSTM autoencoder
        # use the padded dimension
        autoencoder2 = AutoencoderLSTM(n_timesteps_padded, n_features)
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        autoencoder2.compile(optimizer=optimizer, loss="mse")

        # Define the early stopping criteria
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=5, restore_best_weights=True
        )
        # Train the model
        reset_seeds()
        logger.info("Training log for LSTM autoencoder:")
        autoencoder_history2 = autoencoder2.fit(
            X_train_processed_padded,
            X_train_processed_padded,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=True,
            validation_data=(X_test_processed_padded, X_test_processed_padded),
            callbacks=[early_stopping],
        )

        ae_val_loss2 = np.min(autoencoder_history2.history["val_loss"])
        logger.info(f"LSTM autoencoder trained, with validation loss: {ae_val_loss2}.")

        logger.info(
            f"The current prediction margin weight is {pred_margin_weight}, for [LSTM autoencoder]."
        )

        lr_list2 = [A.lr_list[1]] if A.lr_list[1] is not None else [0.001, 0.0001]
        best_lr2, best_cf_model2, best_cf_samples2, _ = find_best_lr(
            classifier,
            X_samples=rand_X_test,
            pred_labels=rand_y_pred,
            autoencoder=autoencoder2,
            lr_list=lr_list2,
            pred_margin_weight=pred_margin_weight,
            step_weights=step_weights,
            random_state=RANDOM_STATE,
            padding_size=padding_size,
            target_prob=A.tau_value,
        )
        logger.info(f"The best learning rate found is {best_lr2}.")

        # ### Evaluation metrics
        # predicted probabilities of CFs
        z_pred2 = classifier.predict(best_cf_samples2)
        cf_pred_labels2 = np.argmax(z_pred2, axis=1)

        # remove extra paddings after counterfactual generation in 1dCNN autoencoder
        best_cf_samples2 = remove_paddings(best_cf_samples2, padding_size)

        evaluate_res2 = evaluate(
            rand_X_test_original,
            best_cf_samples2,
            rand_y_pred,
            cf_pred_labels2,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "LSTM autoencoder",
            acc,
            ae_val_loss2,
            best_lr2,
            evaluate_res2,
            pred_margin_weight=pred_margin_weight,
            step_weight_type=A.w_type.lower(),
            threshold_tau=A.tau_value,
        )
        logger.info(
            f"Done for CF search [LSTM autoencoder], pred_margin_weight={pred_margin_weight}."
        )

        ###############################################
        # ## 2.3 CF search with no autoencoder
        ###############################################
        logger.info(
            f"The current prediction margin weight is {pred_margin_weight}, for [No autoencoder]."
        )

        lr_list3 = [A.lr_list[2]] if A.lr_list[2] is not None else [0.001, 0.0001]
        best_lr3, best_cf_model3, best_cf_samples3, _ = find_best_lr(
            classifier,
            X_samples=rand_X_test,
            pred_labels=rand_y_pred,
            autoencoder=None,
            lr_list=lr_list3,
            pred_margin_weight=pred_margin_weight,
            step_weights=step_weights,
            random_state=RANDOM_STATE,
            padding_size=padding_size,
            target_prob=A.tau_value,
        )
        logger.info(f"The best learning rate found is {best_lr3}.")

        # ### Evaluation metrics
        # predicted probabilities of CFs
        z_pred3 = classifier.predict(best_cf_samples3)
        cf_pred_labels3 = np.argmax(z_pred3, axis=1)

        # remove extra paddings after counterfactual generation in 1dCNN autoencoder
        best_cf_samples3 = remove_paddings(best_cf_samples3, padding_size)

        evaluate_res3 = evaluate(
            rand_X_test_original,
            best_cf_samples3,
            rand_y_pred,
            cf_pred_labels3,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "No autoencoder",
            acc,
            0,
            best_lr3,
            evaluate_res3,
            pred_margin_weight=pred_margin_weight,
            step_weight_type=A.w_type.lower(),
            threshold_tau=A.tau_value,
        )
        logger.info(
            f"Done for CF search [No autoencoder], pred_margin_weight={pred_margin_weight}."
        )
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()
