#!/usr/bin/env python
# coding: utf-8
import logging
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import counterfactuals

from help_functions import (
    ResultWriter,
    evaluate,
    reset_seeds,
    time_series_normalize,
    upsample_minority,
    fit_evaluation_models,
)
from keras_models import *

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
    A = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    RANDOM_STATE = 39

    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)
    logger.info(f"Result writer is ready, writing to {A.output}...")
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

        fold_idx += 1
        logger.info(
            f"Current CV fold: [{fold_idx}], with `X_train` & `X_test` shape: {X_train.shape} | {X_test.shape}."
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
            logger.info(f"Current distribution of y: \n{pd.value_counts(y_train)}.")

        nb_classes = len(np.unique(y_train))
        y_train_classes, y_test_classes = (
            y_train.copy(),
            y_test.copy(),
        )
        y_train, y_test = (
            to_categorical(y_train, nb_classes),
            to_categorical(y_test, nb_classes),
        )

        # ### 1.1 Normalization - fit scaler using training data
        n_training, n_timesteps = X_train.shape
        n_features = 1

        # Reshape to 3-d for training deep learning models
        X_train = X_train.reshape(-1, n_timesteps, n_features)
        X_test = X_test.reshape(-1, n_timesteps, n_features)

        X_train_processed, trained_scaler = time_series_normalize(
            data=X_train, n_timesteps=n_timesteps
        )
        X_test_processed, _ = time_series_normalize(
            data=X_test, n_timesteps=n_timesteps, scaler=trained_scaler
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

        # ## 2. Native Guide CF generation
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds()

        ###############################################
        # ## 2.0 FCN classifier
        ###############################################
        input_shape = X_train.shape[1:]
        classifier_fcn = Classifier_FCN(input_shape, nb_classes)

        classifier_fcn.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        )
        early_stopping_loss = keras.callbacks.EarlyStopping(
            monitor="loss", patience=30, restore_best_weights=True
        )

        batch_size = 16
        nb_epochs = 2000
        mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

        # Train the model
        reset_seeds()
        logger.info("Training FCN classifier...")
        classifier_history = classifier_fcn.fit(
            X_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=nb_epochs,
            # verbose=True, # uncomment for debugging
            verbose=False,
            callbacks=[reduce_lr, early_stopping_loss],
        )

        y_pred = classifier_fcn.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        acc = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes)
        logger.info(f"FCN classifier trained, with test accuracy {acc}.")

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(
                y_true=y_test_classes, y_pred=y_pred_classes, labels=[1, 0]
            ),
            index=["True:pos", "True:neg"],
            columns=["Pred:pos", "Pred:neg"],
        )
        logger.info(f"Confusion matrix: \n{confusion_matrix_df}.")

        ###############################################
        # ## 2.1 Native Guide CF generation
        ###############################################

        # Retrieve CAM weights
        training_weights = get_training_weights(X_train, model=classifier_fcn)

        # finding the nearest unlike neighbour. NB will need to account for regularization
        def native_guide_retrieval(query, predicted_label, distance, n_neighbors):
            df = pd.DataFrame(y_train_classes, columns=["label"])
            df.index.name = "index"
            # df[df['label'] == 1].index.values, df[df['label'] != 1].index.values

            ts_length = X_train.shape[1]

            #     knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric=distance)

            target_nns = df[df["label"] != predicted_label]
            knn.fit(np.squeeze(X_train[list(target_nns.index.values)]))

            dist, ind = knn.kneighbors(
                query.reshape(1, ts_length), return_distance=True
            )
            return dist[0], target_nns.index[ind[0][:]]

        nuns = []
        for instance in range(len(X_test)):
            nuns.append(
                native_guide_retrieval(
                    X_test[instance], y_pred_classes[instance], "euclidean", 1
                )[1][0]
            )
        nuns = np.array(nuns)

        def counterfactual_generator_swap(
            X_input, nun_idx, subarray_length, model, target_label
        ):
            most_influential_array = findSubarray(
                (training_weights[nun_idx]), subarray_length
            )
            starting_point = np.where(
                training_weights[nun_idx] == most_influential_array[0]
            )[0][0]
            X_example = np.concatenate(
                (
                    X_input[:starting_point],
                    X_train[nun_idx][starting_point : subarray_length + starting_point],
                    X_input[subarray_length + starting_point :],
                )
            )
            prob_target = model.predict(X_example.reshape(1, -1, 1))[0][target_label]
            # # uncomment for debugging
            # print(f"Initialized, with prediction probability: {prob_target}.")

            n_timesteps = X_example.shape[0]
            while prob_target < 0.5 and subarray_length < n_timesteps:
                subarray_length += 1

                most_influential_array = findSubarray(
                    (training_weights[nun_idx]), subarray_length
                )
                starting_point = np.where(
                    training_weights[nun_idx] == most_influential_array[0]
                )[0][0]
                X_example = np.concatenate(
                    (
                        X_input[:starting_point],
                        X_train[nun_idx][
                            starting_point : subarray_length + starting_point
                        ],
                        X_input[subarray_length + starting_point :],
                    )
                )
                prob_target = model.predict(X_example.reshape(1, -1, 1))[0][
                    target_label
                ]
                # # Uncomment below for debugging
                # print(
                #     f"Iter:{subarray_length}, with prediction probability: {prob_target}."
                # )

            # # uncomment for debugging
            # print(
            #     f"Finished, with subarray/total length: {subarray_length}/{n_timesteps}, prediction probability: {prob_target}."
            # )

            return X_example

        # used to find the maximum contiguous subarray of length k in the explanation weight vector
        def findSubarray(a, k):
            n = len(a)
            vec = []

            # Iterate to find all the sub-arrays
            for i in range(n - k + 1):
                temp = []

                # Store the sub-array elements in the array
                for j in range(i, i + k):
                    temp.append(a[j])

                # Push the vector in the container
                vec.append(temp)

            sum_arr = []
            for v in vec:
                sum_arr.append(np.sum(v))

            return vec[np.argmax(sum_arr)]

        ### Evaluation metrics
        # Get these instances for CF evaluation; class abnormal (0) VS normal class (1)
        rand_X_test = X_test[rand_test_idx]
        rand_y_pred = y_pred_classes[rand_test_idx]
        rand_nuns = nuns[rand_test_idx]

        cf_cam_swap = []
        n_iter = 0
        for test_instance, nun_idx, pred in zip(rand_X_test, rand_nuns, rand_y_pred):
            target_label = 1 - pred  # for binary classification
            # # uncomment for debugging
            # print(f"Sample: {n_iter}, target label: {target_label}.")
            n_iter += 1
            cf_cam_swap.append(
                counterfactual_generator_swap(
                    test_instance,
                    nun_idx,
                    1,
                    model=classifier_fcn,
                    target_label=target_label,
                )
            )
        print(f"#Sample: {n_iter} finished, in total.")

        # predicted probabilities of CFs
        cf_cam_swap = np.array(cf_cam_swap)
        z_pred = classifier_fcn.predict(cf_cam_swap)
        cf_pred_labels = np.argmax(z_pred, axis=1)

        # normalize negative predicted samples and CF samples before evaluation
        rand_X_test, _ = time_series_normalize(
            data=rand_X_test, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        cf_samples, _ = time_series_normalize(
            data=cf_cam_swap, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        evaluate_res = evaluate(
            np.squeeze(rand_X_test),
            np.squeeze(cf_samples),
            rand_y_pred,
            cf_pred_labels,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "CF - Native Guide",
            acc,
            0,
            0,
            evaluate_res,
            pred_margin_weight=0,
            step_weight_type=0,
            threshold_tau=0,
        )
        logger.info(f"Done for CF search [Native Guide].")

        ##########################################################
        # ## 3. Shapelet forest classifier
        ##########################################################
        # Reshape to 2-d for training RSF & kNN models
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

        shapelet_clf = ShapeletForestClassifier(
            n_shapelets=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
            n_estimators=50,
            max_depth=5,
        )
        # y should be a 1d array in .fit()
        shapelet_clf.fit(X_train, y_train_classes)

        # warnings.filterwarnings(
        #     "ignore", category=FutureWarning
        # )  # ignore warnings of package version
        y_pred_classes2 = shapelet_clf.predict(X_test)

        acc2 = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes2)
        logger.info(f"Shapelet forest classifier trained, with test accuracy {acc2}.")

        # Get these instances of negative predictions, which is class 0
        rand_X_test2 = X_train[rand_test_idx]
        rand_y_pred2 = y_pred_classes2[rand_test_idx]
        desired_labels2 = 1 - rand_y_pred2

        cf_samples2, _, _ = counterfactuals(
            shapelet_clf,
            rand_X_test2,
            desired_labels2,
            scoring="euclidean",
            random_state=RANDOM_STATE,
        )

        # ### Evaluation metrics
        z_pred2 = shapelet_clf.predict_proba(cf_samples2)
        cf_pred_labels2 = np.argmax(z_pred2, axis=1)

        rand_X_test2, _ = time_series_normalize(
            data=rand_X_test2, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        cf_samples2, _ = time_series_normalize(
            data=cf_samples2, n_timesteps=n_timesteps, scaler=trained_scaler
        )

        evaluate_res2 = evaluate(
            np.squeeze(rand_X_test2),
            np.squeeze(cf_samples2),
            rand_y_pred2,
            cf_pred_labels2,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "CF - Shapelet Forest",
            acc2,
            0,
            0,
            evaluate_res2,
            pred_margin_weight=0,
            step_weight_type=0,
            threshold_tau=0,
        )
        logger.info(f"Done for CF search [Shapelet].")

        ##########################################################
        # ## 4. kNN counterfactuals
        ##########################################################
        knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn_clf.fit(X_train, y_train_classes)

        y_pred_classes3 = knn_clf.predict(X_test)
        acc3 = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes3)
        logger.info(f"k-NN classifier trained, with test accuracy {acc3}.")

        # Get the test instances
        rand_X_test3 = X_train[rand_test_idx]
        rand_y_pred3 = y_pred_classes3[rand_test_idx]
        desired_labels3 = 1 - rand_y_pred3

        cf_samples3, _, _ = counterfactuals(
            knn_clf,
            rand_X_test3,
            desired_labels3,
            scoring="euclidean",
            random_state=RANDOM_STATE,
        )

        # ### Evaluation metrics
        z_pred3 = knn_clf.predict_proba(cf_samples3)
        cf_pred_labels3 = np.argmax(z_pred3, axis=1)

        rand_X_test3, _ = time_series_normalize(
            data=rand_X_test3, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        cf_samples3, _ = time_series_normalize(
            data=cf_samples3, n_timesteps=n_timesteps, scaler=trained_scaler
        )

        evaluate_res3 = evaluate(
            np.squeeze(rand_X_test3),
            np.squeeze(cf_samples3),
            rand_y_pred3,
            cf_pred_labels3,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        result_writer.write_result(
            fold_idx,
            "CF - kNN",
            acc3,
            0,
            0,
            evaluate_res3,
            pred_margin_weight=0,
            step_weight_type=0,
            threshold_tau=0,
        )
        logger.info(f"Done for CF search [k-NN].")

    logger.info("Done.")


def get_training_weights(X_train, model):
    w_k_c = model.layers[-1].get_weights()[0]
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]
    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    weights = []
    for i, ts in enumerate(X_train):
        ts = ts.reshape(1, -1, 1)
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted)

        cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
        for k, w in enumerate(w_k_c[:, pred_label]):
            cas += w * conv_out[0, :, k]
        weights.append(cas)

    return np.array(weights)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()
