from tensorflow import keras

"""
1dCNN models
"""
def Autoencoder(n_timesteps, n_features):
    # Define encoder and decoder structure
    def Encoder(input):
        x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(input)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        return x

    def Decoder(input):
        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(input)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
        #x = keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(filters=1, kernel_size=3, activation="linear", padding="same")(x)
        return x

    # Define the AE model
    orig_input = keras.Input(shape=(n_timesteps, n_features))
    autoencoder = keras.Model(inputs=orig_input, outputs=Decoder(Encoder(orig_input)))
    
    return autoencoder

def Classifier(n_timesteps, n_features, n_conv_layers=1, add_dense_layer=True, n_output=1):
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    inputs = keras.Input(shape=(n_timesteps, n_features), dtype="float32")
    
    if add_dense_layer:
        x = keras.layers.Dense(128)(inputs)
    else: 
        x = inputs
    
    for i in range(n_conv_layers):
        x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Flatten()(x)

    if n_output>=2:
        outputs = keras.layers.Dense(n_output, activation='softmax')(x)
    else:
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    classifier = keras.Model(inputs, outputs)
    classifier = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return classifier

"""
LSTM models
"""
def AutoencoderLSTM(n_timesteps, n_features):
    # Define encoder and decoder structure
    # structure from medium post: https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    def EncoderLSTM(input):    
        # x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(input)
        x = keras.layers.LSTM(64, activation='tanh', return_sequences=True)(input)
        # encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(x)
        encoded = keras.layers.LSTM(32, activation='tanh', return_sequences=False)(x)
        return encoded

    def DecoderLSTM(encoded):
        x = keras.layers.RepeatVector(n_timesteps)(encoded)
        # x = keras.layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)
        # x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(n_features, activation='sigmoid'))(x)
        return decoded

    # Define the AE model
    orig_input2 = keras.Input(shape=(n_timesteps, n_features))

    autoencoder2 = keras.Model(inputs=orig_input2, outputs=DecoderLSTM(EncoderLSTM(orig_input2)))
    
    return autoencoder2

def ClassifierLSTM(n_timesteps, n_features, extra_lstm_layer=True, n_output=1):
    # Define the model structure - only LSTM layers
    # https://www.kaggle.com/szaitseff/classification-of-time-series-with-lstm-rnn
    inputs = keras.Input(shape=(n_timesteps, n_features), dtype="float32")
    if extra_lstm_layer:
        x = keras.layers.LSTM(64, activation='tanh', return_sequences=True)(inputs) # set return_sequences true to feed next LSTM layer
    else:
        x = keras.layers.LSTM(32, activation='tanh', return_sequences=False)(inputs) # set return_sequences false to feed dense layer directly
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    if extra_lstm_layer:
        x = keras.layers.LSTM(16, activation='tanh', return_sequences=False)(x)
        x = keras.layers.BatchNormalization()(x)
    
    if n_output>=2:
        outputs = keras.layers.Dense(n_output, activation='softmax')(x)
    else:
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    classifier2 = keras.Model(inputs, outputs)
    
    return classifier2

"""
composite autoencoder
"""

def CompositeAutoencoder(n_timesteps, n_features):
    # https://machinelearningmastery.com/lstm-autoencoders/
    # https://stackoverflow.com/questions/48603328/how-do-i-split-an-convolutional-autoencoder

    # Composite 1dCNN Autoencoder
    # encoder:
    inputs = keras.layers.Input(shape=(n_timesteps, n_features))
    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(inputs)
    x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    encoded = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)

    encoder3 = keras.models.Model(inputs, encoded)

    _, encoded_dim1, encoded_dim2 = encoder3.layers[-1].output_shape

    # decoder
    decoder_inputs = keras.layers.Input(shape=(encoded_dim1, encoded_dim2)) 
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(decoder_inputs)
    x = keras.layers.UpSampling1D(size=2)(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    # x = keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
    x = keras.layers.UpSampling1D(size=2)(x)
    decoded = keras.layers.Conv1D(filters=1, kernel_size=3, activation="linear", padding="same")(x)

    decoder3 = keras.models.Model(decoder_inputs, decoded, name="only_decoder")

    # classifier based on previous encoder
    classifier_inputs = keras.layers.Input(shape=(encoded_dim1, encoded_dim2)) 
    x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding="same")(classifier_inputs)
    x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    classifier_outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    classifier3 = keras.models.Model(classifier_inputs, classifier_outputs, name="only_classifier")

    # composite autoencoder
    encoded = encoder3(inputs)
    decoded = decoder3(encoded)
    classified = classifier3(encoded)

    composite_autoencoder = keras.models.Model(inputs, [decoded, classified])
    
    return composite_autoencoder, encoder3, decoder3, classifier3