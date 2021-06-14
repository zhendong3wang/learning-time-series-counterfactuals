import numpy as np
import tensorflow as tf
from tensorflow import keras

class ModifiedLatentCF:
    """Explanations by generating a counterfacutal sample in the latent space of
    any autoencoder.

    References
    ----------
    Latent-CF: A Simple Baseline for Reverse Counterfactual Explanation
        Rachana Balasubramanian and Samuel Sharpe and Brian Barr and Jason Wittenbach and C. Bayan Brus
        In Proceedings of the Conference on Neural Information Processing Systems, 2020
    """

    def __init__(    
        self, 
        probability=0.5, 
        *, 
        alpha=0.001, 
        tolerance=1e-6, 
        learning_rate=1e-3, 
        max_iter=100,
        optimizer=None,
        autoencoder=None,
        only_encoder=None,
        only_decoder=None
    ):
        """
        Parameters
        ----------
        probability : float, optional
            The desired probability assigned by the model

        alpha : float, optional
            The step size

        tolerance : float, optional
            The maximum difference between the desired and assigned probability

        learning_rate : float, optional
            The learning rate of the optimizer

        max_iter : int, optional
            The maximum number of iterations

        autoencoder : int, optional
            The autoencoder for the latent representation

            - if None the sample is generated in the original space
            - if given, the autoencoder is expected to have `k` decoder layer and `k`
              encoding layers.
        """
        self.optimizer_ = tf.optimizers.Adam(learning_rate=1e-4) if optimizer is None else optimizer
#         self.optimizer_ = tf.optimizers.SGD(learning_rate=0.05, momentum=0.9) # uncomment if need an optimizer with no acceleration on gradients from previous iterations
        self.loss_ = keras.losses.MeanSquaredError() # TODO: allow different loss functions
        self.alpha_ = tf.constant(alpha)
        self.probability_ = tf.constant(probability)
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        self.autoencoder = autoencoder
        self.only_encoder = only_encoder
        self.only_decoder = only_decoder
    
    def fit(self, model):
        """Fit a new counterfactual explainer to the model

        Paramaters
        ----------

        model : keras.Model
            The model
        """
        if self.autoencoder:
            encode_input, encode_output, decode_input, decode_output = extract_encoder_decoder(self.autoencoder)
            self.decoder_ = keras.Model(inputs=decode_input, outputs=decode_output)
            self.encoder_ = keras.Model(inputs=encode_input, outputs=encode_output)
        elif self.only_encoder and self.only_decoder:
            self.encoder_ = self.only_encoder
            self.decoder_ = self.only_decoder
        else:
            self.decoder_ = None
            self.encoder_ = None
        self.model_ = model
        return self

    def predict(self, x):
        """Compute the differnece beteween the desired and actual probability

        Paramters
        ---------
        x : Variable
            Variable of the sample
        """
        if self.autoencoder is None:
            z = x
        else:
            z = self.decoder_(x)
        
        return self.model_(z)
    
    def compute_loss(self, prediction):
    
        return self.loss_(self.probability_, prediction)

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        if self.only_encoder is not None: # if only encoder, then return the latent embeddings
            _, encoded_dim1, encoded_dim2 = self.only_encoder.layers[-1].output_shape
            result_samples = np.empty((x.shape[0], encoded_dim1, encoded_dim2)) 
        else: 
            result_samples = np.empty(x.shape)

        losses = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            if i % 50 == 0: print(f'{i} samples been transformed.')

            x_sample, loss = self._transform_sample(x[np.newaxis, i])
            result_samples[i] = x_sample
            losses[i] = loss

        return result_samples, losses
    
    def _transform_sample(self, x):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # TODO: check_is_fitted(self)
        if self.only_encoder or self.autoencoder is not None:
            z = tf.Variable(self.encoder_(x))
        else:
            z = tf.Variable(x, dtype=tf.float32)
        
        pred = self.predict(z)
        loss = self.compute_loss(pred)
        it = 0
        
        with tf.GradientTape() as tape:
#             z.assign(z - self.alpha_ * loss)
            pred = self.predict(z)
#             # uncomment for debugging
#             print(f'predicted probability {pred}')
            loss = self.compute_loss(pred)
        
        while (loss > self.tolerance_ or pred < self.probability_) and (it < self.max_iter if self.max_iter else True):
#             # uncomment for debugging
#             print(f'current loss {loss:.5f}, iteration {it}')

            # Get gradients of loss wrt the sample
            grads = tape.gradient(loss, z)
            # Update the weights of the sample
            self.optimizer_.apply_gradients([(grads, z)])
            
            with tf.GradientTape() as tape:
#                 z.assign(z - self.alpha_ * loss)
                pred = self.predict(z)
#                 # uncomment for debugging
#                 print(f'predicted probability {pred}')
                loss = self.compute_loss(pred)
            
            it += 1

#         # uncomment for debugging
#         predicted = self.model_(z)
#         print(f'predicted probability before returning {predicted}, iteration {it}')

        return z.numpy() if self.autoencoder is None else self.decoder_(z).numpy(), float(loss)

def extract_encoder_decoder(autoencoder):
    """Extract the encoder and decoder from an autoencoder

    autoencoder : keras.Model
        The autoencoder of `k` encoders and `k` decoders
    """
    depth = len(autoencoder.layers) // 2    
    encoder = autoencoder.layers[1](autoencoder.input)
    for i in range(2, depth):
        encoder = autoencoder.layers[i](encoder)

    encode_input = keras.Input(shape=encoder.shape[1:])
    decoder = autoencoder.layers[depth](encode_input)
    for i in range(depth + 1, len(autoencoder.layers)):
        decoder = autoencoder.layers[i](decoder)

    return autoencoder.input, encoder, encode_input, decoder