#!/usr/bin/env python
# coding: utf-8
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from wildboar.datasets import load_dataset

from help_functions import (ResultWriter, conditional_pad, evaluate,
                            find_best_lr, reset_seeds, time_series_normalize,
                            upsample_minority)
from keras_models import *

os.environ['TF_DETERMINISTIC_OPS'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def main():
    parser = ArgumentParser(description='Run this script to evaluate LatentCF method.')
    parser.add_argument('--dataset', type=str, help='Dataset that the experiment is running on.') 
    parser.add_argument('--pos', type=int, default=1, help='The positive label of the dataset, e.g. 1 or 2.') 
    parser.add_argument('--neg', type=int, default=0, help='The negative label of the dataset, e.g. 0 or -1') 
    parser.add_argument('--output', type=str, help='Output file name.')
    parser.add_argument('--shallow-cnn', action="store_true", default=False, help='Boolean flag of using (relatively) shallower CNN model, default False.')     
    parser.add_argument('--shallow-lstm', action="store_true", default=False, help='Boolean flag of using (relatively) shallower LSTM model, default False.')     
    A = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    RANDOM_STATE = 39
    PRED_MARGIN_W_LIST = [1.0, 0.75, 0.5, 0.25, 0.0]

    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)
    logger.info(f"Result writer is ready, writing to {A.output}...")
    result_writer.write_head()

    # 1. Load data
    X, y = load_dataset(A.dataset, repository="wildboar/ucr")

    # Convert positive and negative labels to 1 and 0
    pos_label, neg_label = 1, 0
    y_copy = y.copy()
    if A.pos != pos_label:
        y_copy[y==A.pos] = pos_label # convert/normalize positive label to 1
    if A.neg != neg_label:
        y_copy[y==A.neg] = neg_label # convert negative label to 0

    X_train, X_test, y_train, y_test = train_test_split(X, y_copy, test_size=0.2, random_state=RANDOM_STATE, stratify=y_copy) 
        
    # Upsample the minority class
    y_train_copy = y_train.copy()
    X_train, y_train = upsample_minority(X_train, y_train, pos_label=pos_label, neg_label=neg_label)
    if y_train.shape != y_train_copy.shape:
        logger.info(f"Data upsampling performed, current distribution of y: \n{pd.value_counts(y_train)}.")

    y_train_classes = y_train.copy()
    y_test_classes = y_test.copy()
    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    # ### 1.1 Normalization - fit scaler using training data 
    n_training, n_timesteps = X_train.shape
    n_features = 1

    X_train_processed, trained_scaler = time_series_normalize(data=X_train, n_timesteps=n_timesteps)
    X_test_processed, _ = time_series_normalize(data=X_test, n_timesteps=n_timesteps, scaler=trained_scaler)

    X_train_processed_padded, padding_size = conditional_pad(X_train_processed) # add extra padding zeros if n_timesteps cannot be divided by 4, required for 1dCNN autoencoder structure
    X_test_processed_padded, _ = conditional_pad(X_test_processed) 
    n_timesteps_padded = X_train_processed_padded.shape[1]
    logger.info(f"Data pre-processed, original #timesteps={n_timesteps}, padded #timesteps={n_timesteps_padded}.")

    # ## 2. LatentCF models
    # reset seeds for numpy, tensorflow, python random package and python environment seed
    reset_seeds()

    ###############################################
    # ## 2.1 1dCNN autoencoder + 1dCNN classifier
    ###############################################
    # ### 1dCNN autoencoder
    autoencoder = Autoencoder(n_timesteps_padded, n_features)
    optimizer = keras.optimizers.Adam(lr=0.0005)
    autoencoder.compile(optimizer=optimizer, loss="mse") 

    # Define the early stopping criteria
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)
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
        validation_data=(X_test_processed_padded, X_test_processed_padded),
        callbacks=[early_stopping])

    ae_val_loss = np.min(autoencoder_history.history['val_loss'])
    logger.info(f"1dCNN autoencoder trained, with validation loss: {ae_val_loss}.")

    # ### 1dCNN classifier
    if A.shallow_cnn == True:
        logger.info(f"Check shallow_cnn argument={A.shallow_cnn}, use the shallow structure.")
        classifier = Classifier(n_timesteps_padded, n_features, n_conv_layers=1, add_dense_layer=True, n_output=2) # shallow CNN for small data size
    else:
        classifier = Classifier(n_timesteps_padded, n_features, n_conv_layers=3, add_dense_layer=False, n_output=2) # deeper CNN layers for data with larger size

    optimizer = keras.optimizers.Adam(lr=0.0001)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Define the early stopping criteria
    early_stopping_accuracy = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
    # Train the model
    reset_seeds()
    logger.info("Training log for 1dCNN classifier:")
    classifier_history = classifier.fit(X_train_processed_padded, 
            y_train, 
            epochs=150,
            batch_size=32,
            shuffle=True, 
            verbose=True, 
            validation_data=(X_test_processed_padded, y_test),
            callbacks=[early_stopping_accuracy])

    y_pred = classifier.predict(X_test_processed_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)
    acc = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes)
    logger.info(f"1dCNN classifier trained, with validation accuracy {acc}.")

    ### Evaluation metrics
    # use radius to find the count of points - KDTree; a trained tree is needed for evaluation
    tree = KDTree(
        X_train_processed[y_train_classes==pos_label].reshape(-1, n_timesteps), 
        leaf_size=40, metric='euclidean')
    max_distance = distance_matrix(
        X_train_processed[y_train_classes==neg_label].reshape(-1, n_timesteps), 
        X_train_processed[y_train_classes==pos_label].reshape(-1, n_timesteps)).max()
    
    for pred_margin_weight in PRED_MARGIN_W_LIST:
        # Get these instances of negative predictions, which is class abnormal (0); (normal is class 1)
        X_pred_neg = X_test_processed_padded[y_pred_classes == neg_label]
        # best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
        #     classifier, X_pred_neg, autoencoder=autoencoder, 
        #     pred_margin_weight=pred_margin_weight, step_weight_type='local', random_state=RANDOM_STATE)
        best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
            classifier, X_pred_neg, autoencoder=autoencoder, lr_list=[0.0001], 
            pred_margin_weight=pred_margin_weight, step_weight_type='local', random_state=RANDOM_STATE)
        logger.info(f"The best learning rate found is {best_lr}.")

        z_pred = classifier.predict(best_cf_samples)[:, 1] # predicted probabilities of CFs
        if padding_size != 0:
            best_cf_samples = best_cf_samples[:, :-padding_size, :] # remove extra paddings after counterfactual generation
            X_pred_neg_orignal = X_test_processed[y_pred_classes == neg_label] # use the unpadded X for evaluation
        else:
            X_pred_neg_orignal = X_pred_neg
        
        evaluate_res = evaluate(X_pred_neg_orignal, best_cf_samples, z_pred, n_timesteps, tree, max_distance)

        result_writer.write_result("modified: 1dCNN classifier + 1dCNN autoencoder", acc, ae_val_loss, best_lr, evaluate_res, pred_margin_weight=pred_margin_weight, step_weight_type='local')
        logger.info(f"Done for 1dCNN classifier + 1dCNN autoencoder, modified LatentCF, pred_margin_weight={pred_margin_weight}.")

    ###############################################
    # ## 2.2 LSTM autoencoder + LSTM classifier
    ###############################################
    # ### LSTM autoencoder
    autoencoder2 = AutoencoderLSTM(n_timesteps, n_features) # use the non-padded dimension
    optimizer = keras.optimizers.Adam(lr=0.0001)
    autoencoder2.compile(optimizer=optimizer, loss="mse") 

    # Define the early stopping criteria
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True) # modify patience (from 10) to 5
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
    # Train the model
    reset_seeds()
    logger.info("Training log for LSTM autoencoder:")
    autoencoder_history2 = autoencoder2.fit(
        X_train_processed, 
        X_train_processed, 
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=True, 
        validation_data=(X_test_processed, X_test_processed),
        callbacks=[early_stopping])

    ae_val_loss2 = np.min(autoencoder_history2.history['val_loss'])
    logger.info(f"LSTM autoencoder trained, with validation loss: {ae_val_loss2}.")

    # ### LSTM classifier
    if A.shallow_lstm == True:
        logger.info(f"Check shallow_lstm argument={A.shallow_lstm}, use the shallow structure.")
        classifier2 = ClassifierLSTM(n_timesteps, n_features, extra_lstm_layer=False, n_output=2) # shallow LSTM for small data size
    else:
        classifier2 = ClassifierLSTM(n_timesteps, n_features, extra_lstm_layer=True, n_output=2) 
    optimizer = keras.optimizers.Adam(lr=0.0005)
    classifier2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Define the early stopping criteria
    early_stopping_accuracy = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True) # patient = 5 or 10 doesn't guarantee find an optimal
    # Train the model
    reset_seeds()
    logger.info("Training log for LSTM classifier:")
    classifier_history2 = classifier2.fit(X_train_processed, 
            y_train, 
            epochs=150,
            batch_size=32,
            shuffle=True, 
            verbose=True, 
            validation_data=(X_test_processed, y_test),
            callbacks=[early_stopping_accuracy])

    y_pred2 = classifier2.predict(X_test_processed)
    y_pred_classes2 = np.argmax(y_pred2, axis=1)
    acc2 = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes2)
    logger.info(f"LSTM classifier trained, with validation accuracy {acc2}.")

    for pred_margin_weight in PRED_MARGIN_W_LIST:     
        # Get these instances of negative predictions
        X_pred_neg2 = X_test_processed[y_pred_classes2 == neg_label]

        # best_lr2, best_cf_model2, best_cf_samples2, _ = find_best_lr(
        #     classifier2, X_pred_neg2, autoencoder=autoencoder2, 
        #     pred_margin_weight=pred_margin_weight, step_weight_type='local', random_state=RANDOM_STATE)
        best_lr2, best_cf_model2, best_cf_samples2, _ = find_best_lr(
            classifier2, X_pred_neg2, autoencoder=autoencoder2, lr_list=[0.0001], 
            pred_margin_weight=pred_margin_weight, step_weight_type='local', random_state=RANDOM_STATE)
        logger.info(f"The best learning rate found is {best_lr2}.")


        # ### Evaluation metrics
        z_pred2 = classifier2.predict(best_cf_samples2)[:, 1] # predicted probabilities of CFs
        evaluate_res2 = evaluate(X_pred_neg2, best_cf_samples2, z_pred2, n_timesteps, tree, max_distance)

        result_writer.write_result("modified: LSTM classifier + LSTM autoencoder", acc2, ae_val_loss2, best_lr2, evaluate_res2, pred_margin_weight=pred_margin_weight, step_weight_type='local')
        logger.info(f"Done for LSTM classifier + LSTM autoencoder, modified LatentCF, pred_margin_weight={pred_margin_weight}.")


    # ##########################################################
    # # ## 2.3 composite: 1dCNN autoencoder + 1dCNN classifier
    # ##########################################################
    # composite_autoencoder, encoder3, decoder3, classifier3 = CompositeAutoencoder(n_timesteps_padded, n_features, n_output_classifier=2)
    # # Define separated loss functions for decoder and classifier
    # composite_autoencoder.compile(
    #     optimizer="adam", 
    #     loss=[
    #         keras.losses.MeanSquaredError(),
    #         keras.losses.BinaryCrossentropy(from_logits=True),
    #     ],
    #     loss_weights=[0.5, 1.0],
    #     metrics=['accuracy']) 

    # # Define the early stopping criteria
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=3, restore_best_weights=True)
    # reset_seeds()
    # logger.info("Training log for 1dCNN composite classifier+autoencoder:")
    # autoencoder_history3 = composite_autoencoder.fit(
    #     X_train_processed_padded, 
    #     [X_train_processed_padded, y_train], 
    #     epochs=50,
    #     batch_size=32,
    #     shuffle=True,
    #     verbose=True, 
    #     validation_data=(X_test_processed_padded, [X_test_processed_padded, y_test]),
    #     callbacks=[early_stopping]) 

    # min_loss_id = np.argmin(autoencoder_history3.history['val_loss'])
    # ae_val_loss3 = autoencoder_history3.history['val_only_decoder_loss'][min_loss_id]

    # X_encoded = encoder3.predict(X_test_processed_padded)
    # # X_decoded = decoder3.predict(X_encoded)
    # y_pred3 = classifier3.predict(X_encoded)
    # y_pred_classes3 = np.argmax(y_pred3, axis=1)
    # acc3 = balanced_accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes3)
    # logger.info(f"1dCNN composite autoencoder + classifier trained, with decoder loss {ae_val_loss3}, validation accuracy {acc3}.")

    # for pred_margin_weight in PRED_MARGIN_W_LIST:     
    #     # Get these instances of negative predictions
    #     X_pred_neg3 = X_test_processed_padded[y_pred_classes3 == neg_label]

    #     best_lr3, best_cf_model3, best_cf_samples3, best_cf_embeddings3 = find_best_lr(classifier3, X_pred_neg3, encoder=encoder3, decoder=decoder3, lr_list=[0.001], pred_margin_weight=pred_margin_weight, step_weight_type='local')
    #     logger.info(f"The best learning rate found is {best_lr3}.")

    #     # ### Evaluation metrics
    #     z_pred3 = classifier3.predict(best_cf_embeddings3)[:, 1] # predicted probabilities of CFs
    #     if padding_size != 0:
    #         best_cf_samples3 = best_cf_samples3[:, :-padding_size, :] # remove extra paddings after counterfactual generation
    #         X_pred_neg3_origianl = X_test_processed[y_pred_classes3 == neg_label] # use the unpadded X for evaluation
    #     else: xxx
    #          
    #     evaluate_res3 = evaluate(X_pred_neg3_origianl, best_cf_samples3, z_pred3, n_timesteps, tree, max_distance)

    #     result_writer.write_result("modified: 1dCNN composite classifier + autoencoder", acc3, ae_val_loss3, best_lr3, evaluate_res3, pred_margin_weight=pred_margin_weight, step_weight_type='local')
    #     logger.info(f"Done for 1dCNN composite classifier + autoencoder, pred_margin_weight={pred_margin_weight}.")

    logger.info("Done.")

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    main()
