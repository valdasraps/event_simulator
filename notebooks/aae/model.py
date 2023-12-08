import typing

import numpy as np
from keras import Input, Model
from keras import backend as K
from keras import initializers
from keras.src.layers import Activation, Add, Dense, Lambda

# from keras.src.utils import plot_model


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_vae_edis_encoder_decoder_disc(
    input_dim: int,
    latent_dim: int = 20,
    encoder_dim: int = 128,
    intermediate_dim: int = 128,
) -> typing.Tuple[Model, Model, Model, Model, Model]:
    elu_var: float = np.sqrt(1.55 / intermediate_dim)
    inputs = Input(shape=(input_dim,), name="encoder_input")
    x1 = Dense(
        encoder_dim,
        activation="elu",
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(inputs)
    x2 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x1)
    x2 = Activation("elu")(x2)
    x3 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x2)
    sc1 = Add()([x1, x3])
    x3 = Activation("elu")(sc1)
    x4 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x3)
    sc2 = Add()([x2, x4])
    x4 = Activation("elu")(sc2)
    x5 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x4)
    sc3 = Add()([x3, x5])
    x5 = Activation("elu")(sc3)
    x6 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x5)
    sc4 = Add()([x4, x6])
    x6 = Activation("elu")(sc4)
    z_mean = Dense(latent_dim, name="z_mean")(x6)
    z_log_var = Dense(latent_dim, name="z_log_var")(x6)

    # use re-parameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    # plot_model(encoder, "encoder.png", show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x1 = Dense(
        intermediate_dim,
        activation="elu",
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(latent_inputs)
    x2 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x1)
    x2 = Activation("elu")(x2)
    x3 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x2)
    sc1 = Add()([x1, x3])
    x3 = Activation("elu")(sc1)
    x4 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x3)
    sc2 = Add()([x2, x4])
    x4 = Activation("elu")(sc2)
    x5 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x4)
    sc3 = Add()([x3, x5])
    x5 = Activation("elu")(sc3)
    x6 = Dense(
        encoder_dim,
        kernel_initializer=initializers.random_normal(mean=0.0, stddev=elu_var),
    )(x5)
    sc4 = Add()([x4, x6])
    x6 = Activation("elu")(sc4)
    outputs = Dense(input_dim, activation="tanh")(x6)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name="decoder")
    # decoder.summary()
    # plot_model(decoder, "decoder.png", show_shapes=True)

    # build disc model
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling1")
    x1 = Dense(
        32,
        activation="relu",
    )(latent_inputs)
    x2 = Dense(
        32,
        activation="relu",
    )(x1)
    flag = Dense(1, activation="sigmoid")(x2)

    # instantiate disc model
    disc = Model(latent_inputs, flag, name="discriminant")
    # disc.summary()
    # plot_model(disc, "disc.png", show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name="vae")

    # instantiate EDIS model
    flag = disc(encoder(inputs)[2])
    edis = Model(inputs, flag, name="edis")

    return vae, edis, encoder, decoder, disc
