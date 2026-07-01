import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import tensorflow as tf
from keras import layers, ops
from keras.models import Model


@keras.saving.register_keras_serializable(package='custom_layers')
class Sampling(layers.Layer):
    """
    Custom Keras layer implementing the reparameterization trick for VAE sampling.

    This layer samples from the latent space distribution using the reparameterization
    trick: z = mean + exp(0.5 * log_var) * epsilon, where epsilon ~ N(0, 1).
    This allows backpropagation through the sampling operation during training.

    Attributes:
        seed_generator: Keras random seed generator for reproducible sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = super().get_config()
        return config


class CVAE(Model):
    """
    Convolutional Variational Autoencoder (CVAE) for image data.

    A VAE implementation using convolutional layers for encoding and decoding.
    The model learns a compressed latent representation of input images and can
    reconstruct them. Uses the reparameterization trick for backpropagation through
    the stochastic latent space.

    The loss function consists of:
    - Reconstruction loss: Binary cross-entropy between input and reconstruction
    - KL divergence loss: Regularizes the latent space to approximate N(0, 1)
    - Total loss: reconstruction_loss + beta * kl_loss

    Architecture:
    - Encoder: Strided Conv2D layers -> Flatten -> Dense -> Latent (z_mean, z_log_var)
    - Decoder: Dense -> Reshape -> Conv2DTranspose layers -> Reconstruction

    Attributes:
        input_shape (tuple): Shape of input images (height, width, channels).
        latent_dim (int): Dimensionality of the latent space.
        dense_dim (int): Dimensionality of dense layers.
        conv_layers (tuple): Number of filters in each convolutional layer.
        dropout (float): Dropout rate for regularization.
        beta (float): Weight for KL divergence loss (beta-VAE parameter).
        encoder (Model): Encoder model.
        decoder (Model): Decoder model.

    Example:
        >>> model = CVAE(
        ...     input_shape=(128, 128, 4),
        ...     latent_dim=64,
        ...     dense_dim=256,
        ...     conv_layers=(8, 16, 32, 64, 128),
        ...     dropout=0.25,
        ...     beta=1.0
        ... )
        >>> model.compile(optimizer='adam')
        >>> model.fit(train_data, epochs=100)
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (128, 128, 4),
        latent_dim: int = 128,
        dense_dim: int = 128,
        conv_layers: tuple[int, ...] = (8, 16, 32, 64, 128),
        dropout: float = 0.5,
        beta: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dense_dim = dense_dim
        self.conv_layers = conv_layers
        self.dropout = dropout
        self.beta = beta
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name='reconstruction_loss'
        )
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.total_loss_tracker_val = keras.metrics.Mean(name='total_loss_val')
        self.reconstruction_loss_tracker_val = keras.metrics.Mean(
            name='reconstruction_loss_val'
        )
        self.kl_loss_tracker_val = keras.metrics.Mean(name='kl_loss_val')

    def build_encoder(self) -> keras.Model:
        """
        Build the convolutional encoder network.

        Stacks strided ``Conv2D`` layers (one per entry in ``self.conv_layers``)
        followed by a dense projection, and outputs ``z_mean``, ``z_log_var`` and
        the reparameterized latent sample ``z``.

        Returns:
            keras.Model: Encoder mapping an input patch to ``[z_mean, z_log_var, z]``.
        """
        encoder_inputs = keras.Input(shape=self.input_shape)
        for i, n in enumerate(self.conv_layers):
            if i == 0:
                x = layers.Conv2D(
                    n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
                )(encoder_inputs)
            else:
                x = layers.Conv2D(
                    n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
                )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.dense_dim, activation='relu')(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self) -> keras.Model:
        """
        Build the transposed-convolutional decoder network.

        Projects the latent vector back to a spatial feature map and applies
        stacked ``Conv2DTranspose`` layers (upsampling) to reconstruct all input
        channels.

        Returns:
            keras.Model: Decoder mapping a latent vector to a reconstructed patch.
        """
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        dim_0 = self.input_shape[0] // 2 ** len(self.conv_layers)
        dim_1 = self.input_shape[1] // 2 ** len(self.conv_layers)
        x = layers.Dense(dim_0 * dim_1 * self.conv_layers[0], activation='relu')(
            latent_inputs
        )
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        x = layers.Reshape((dim_0, dim_1, self.conv_layers[0]))(x)

        for n in self.conv_layers[::-1]:
            x = layers.Conv2DTranspose(
                n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
            )(x)

        decoder_outputs = layers.Conv2DTranspose(
            self.input_shape[-1], 3, activation='sigmoid', padding='same'
        )(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.total_loss_tracker_val,
            self.reconstruction_loss_tracker_val,
            self.kl_loss_tracker_val,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            reconstruction_loss *= self.input_shape[-1]
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = ops.mean(
            ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )
        reconstruction_loss *= self.input_shape[-1]
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + (self.beta * kl_loss)
        self.total_loss_tracker_val.update_state(total_loss)
        self.reconstruction_loss_tracker_val.update_state(reconstruction_loss)
        self.kl_loss_tracker_val.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker_val.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker_val.result(),
            'kl_loss': self.kl_loss_tracker_val.result(),
        }


class CondCVAE(CVAE):
    """
    Conditional Convolutional Variational Autoencoder (CondCVAE).

    Extends CVAE to support class-conditional generation. The model conditions both
    the encoder and decoder on one-hot encoded class labels, allowing it to learn
    class-specific latent representations and generate samples conditioned on
    specific classes.

    The conditioning is implemented by concatenating one-hot encoded labels with:
    - Encoder: Concatenated with flattened features before dense layers
    - Decoder: Concatenated with latent vector before dense layers

    Inherits all functionality from CVAE with modified architecture to accept
    conditional inputs.

    Attributes:
        n_classes (int): Number of classes for conditional generation (one-hot dimension).
        All other attributes inherited from CVAE.

    Example:
        >>> model = CondCVAE(
        ...     n_classes=3,
        ...     input_shape=(128, 128, 4),
        ...     latent_dim=64,
        ...     dense_dim=256,
        ...     beta=1.0
        ... )
        >>> model.compile(optimizer='adam')
        >>> # Train with (images, conditions) tuples
        >>> model.fit((train_images, train_conditions), epochs=100)
    """

    def __init__(self, n_classes: int, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)

    def build_encoder(self) -> keras.Model:
        """
        Build the conditional encoder network.

        Like :meth:`CVAE.build_encoder`, but concatenates the one-hot condition
        inputs (``self.n_classes`` wide) with the flattened features before the
        dense projection, so the latent space is conditioned on the metadata.

        Returns:
            keras.Model: Encoder mapping ``[patch, condition]`` to
                ``[z_mean, z_log_var, z]``.
        """
        encoder_inputs = keras.Input(shape=self.input_shape)
        condition_inputs = keras.Input(shape=(self.n_classes,))
        for i, n in enumerate(self.conv_layers):
            if i == 0:
                x = layers.Conv2D(
                    n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
                )(encoder_inputs)
            else:
                x = layers.Conv2D(
                    n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
                )(x)

        x = layers.Flatten()(x)
        x = layers.concatenate([x, condition_inputs])
        x = layers.Dense(self.dense_dim, activation='relu')(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(
            [encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name='encoder'
        )
        return encoder

    def build_decoder(self) -> keras.Model:
        """
        Build the conditional decoder network.

        Like :meth:`CVAE.build_decoder`, but concatenates the one-hot condition
        inputs (``self.n_classes`` wide) with the latent vector before decoding.

        Returns:
            keras.Model: Decoder mapping ``[latent, condition]`` to a
                reconstructed patch.
        """
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        condition_inputs = keras.Input(shape=(self.n_classes,))
        x = layers.concatenate([latent_inputs, condition_inputs])
        dim_0 = self.input_shape[0] // 2 ** len(self.conv_layers)
        dim_1 = self.input_shape[1] // 2 ** len(self.conv_layers)
        x = layers.Dense(dim_0 * dim_1 * self.conv_layers[0], activation='relu')(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        x = layers.Reshape((dim_0, dim_1, self.conv_layers[0]))(x)

        for n in self.conv_layers[::-1]:
            x = layers.Conv2DTranspose(
                n, 3, activation='relu', strides=2, padding='same', name=f'conv_{n}'
            )(x)

        decoder_outputs = layers.Conv2DTranspose(
            self.input_shape[-1], 3, activation='sigmoid', padding='same'
        )(x)
        decoder = keras.Model(
            [latent_inputs, condition_inputs], decoder_outputs, name='decoder'
        )
        return decoder

    @tf.function
    def train_step(self, data):
        data, condition = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([data, condition])
            reconstruction = self.decoder([z, condition])
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            reconstruction_loss *= self.input_shape[-1]
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        data, condition = data
        z_mean, z_log_var, z = self.encoder([data, condition])
        reconstruction = self.decoder([z, condition])
        reconstruction_loss = ops.mean(
            ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )
        reconstruction_loss *= self.input_shape[-1]
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + (self.beta * kl_loss)
        self.total_loss_tracker_val.update_state(total_loss)
        self.reconstruction_loss_tracker_val.update_state(reconstruction_loss)
        self.kl_loss_tracker_val.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker_val.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker_val.result(),
            'kl_loss': self.kl_loss_tracker_val.result(),
        }
