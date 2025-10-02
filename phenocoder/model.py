import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras.models import Model


@keras.saving.register_keras_serializable(package='custom_layers')
class Sampling(layers.Layer):
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
    def __init__(
        self,
        input_shape=(128, 128, 4),
        latent_dim=128,
        dense_dim=128,
        conv_layers=(8, 16, 32, 64, 128),
        dropout=0.5,
        beta=1,
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

    def build_encoder(self):
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

    def build_decoder(self):
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
    def __init__(self, n_classes=2, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)

    def build_encoder(self):
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

    def build_decoder(self):
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
