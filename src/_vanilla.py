import numpy as np
import tensorflow as tf
from tensorflow import keras

class LatentCF:
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
        tolerance=1e-5, 
        learning_rate=1e-3, 
        max_iter=100,
        autoencoder=None,
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
        # self.optimizer_ = tf.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
        # self.optimizer_ = tf.optimizers.Adam(learning_rate=1e-4)
        self.loss_ = keras.losses.MeanSquaredError() # TODO: allow different loss functions
        self.alpha_ = tf.constant(alpha)
        self.probability_ = tf.constant(probability)
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        self.autoencoder = autoencoder
    
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
        else:
            self.decoder_ = None
            self.encoder_ = None
        self.model_ = model
        return self

    def compute_loss(self, x):
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

        predicted = self.model_(z)
#         # uncomment for debugging
#         print(f'predicted probability {predicted}')
        
        return self.loss_(self.probability_, predicted)
    
    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        x_samples = np.empty(x.shape)
        losses = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            if i % 50 == 0: print(f'{i} samples been transformed.')

            x_sample, loss = self._transform_sample(x[np.newaxis, i])
            x_samples[i] = x_sample
            losses[i] = loss

        return x_samples, losses
    
    def _transform_sample(self, x):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # TODO: check_is_fitted(self)
        if self.autoencoder is not None:
            z = tf.Variable(self.encoder_(x))
        else:
            z = tf.Variable(x, dtype=tf.float32)
        
        # calculate loss before starting the 1st iteration
        with tf.GradientTape() as tape:
            loss = self.compute_loss(z)
        it = 0
        
        while loss > self.tolerance_ and (it < self.max_iter if self.max_iter else True):
#             # uncomment for debugging
#             print(f'current loss {loss:.5f}, iteration {it}')
            
            grads = tape.gradient(loss, z)
            z.assign(z - self.alpha_ * grads) # TODO: conflicting with `apply_gradients([(grads, z)])`
#             self.optimizer_.apply_gradients([(grads, z)])

            with tf.GradientTape() as tape:
                loss = self.compute_loss(z)
            
            it += 1

#         # uncomment for debugging
#         predicted = self.model_(self.decoder_(z))
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