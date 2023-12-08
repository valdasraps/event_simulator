import os
from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.src.callbacks import CSVLogger, History, ModelCheckpoint
from keras.src.engine.base_layer import Layer
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.losses import mse
from keras.src.optimizers import SGD  # , Adam
from model import get_vae_encoder_decoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam


def rm_file(file_name: str) -> str:
    try:
        os.remove(file_name)
    except OSError:
        pass
    return file_name


if __name__ == "__main__":
    parser = OptionParser(usage="%prog -d %data_file -w %weights_file")
    parser.add_option(
        "-d", "--data", dest="data", help="Training data file", metavar="data"
    )
    parser.add_option(
        "-w", "--weights", dest="weights", help="Weights file", metavar="weights"
    )

    (options, args) = parser.parse_args()

    if options.data is None:
        parser.error("Data file not provided")
        exit(1)

    ttb_df = pd.read_csv(options.data, sep=" ", header=None)
    data = ttb_df.values.copy()

    max = np.empty(data.shape[1])
    for i in range(0, data.shape[1]):
        max[i] = np.max(np.abs(data[:, i]))
        if np.abs(max[i]) > 0:
            data[:, i] = data[:, i] / max[i]
        else:
            pass

    x_train, x_test = train_test_split(data, test_size=0.25, random_state=25)
    input_dim = x_train.shape[1]

    x_train = np.reshape(x_train, [-1, input_dim]).astype("float32")
    x_test = np.reshape(x_test, [-1, input_dim]).astype("float32")

    epochs = 240
    batch_size = 1024
    latent_dim = 20

    def vae_loss(
        x: Layer,
        x_decoded_mean: Layer,
        z_log_var: Layer,
        z_mean: Layer,
    ) -> KerasTensor:
        mse_loss = mse(x, x_decoded_mean)
        kl_loss = z_log_var + 1 - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        beta = 10 ** (-6)
        loss = K.mean((1 - beta) * mse_loss + beta * kl_loss)
        return loss

    def inv_mass(x: Layer) -> tf.Tensor:
        e_tensor = tf.gather(x, [0, 4, 8, 12], axis=1)
        x_tensor = tf.gather(x, [1, 5, 9, 13], axis=1)
        y_tensor = tf.gather(x, [2, 6, 10, 14], axis=1)
        z_tensor = tf.gather(x, [3, 7, 11, 15], axis=1)
        return (
            tf.square(tf.reduce_sum(e_tensor, axis=1))
            - tf.square(tf.reduce_sum(x_tensor, axis=1))
            - tf.square(tf.reduce_sum(y_tensor, axis=1))
            - tf.square(tf.reduce_sum(z_tensor, axis=1))
        )

    def inv_mass_loss(
        x: Layer,
        x_decoded_mean: Layer,
    ) -> tf.Tensor:
        return tf.square(inv_mass(x) - inv_mass(x_decoded_mean))

    vae, encoder, decoder = get_vae_encoder_decoder(input_dim=input_dim, latent_dim=20)

    learnrate = 0.001
    iterations = 7
    lr_limit = 0.001 / (2**iterations)
    history = History()
    checkpointer = ModelCheckpoint(
        filepath=rm_file(f"{options.data}.weights.hdf5"), verbose=1, save_best_only=True
    )
    logger = CSVLogger(rm_file(f"{options.data}.log"), separator=",", append=True)
    opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    vae.add_loss(inv_mass_loss(x=vae.input, x_decoded_mean=vae.output))
    vae.compile(optimizer=opt, loss=None)

    vae.summary()

    k = 0
    if options.weights:
        vae.load_weights(options.weights)
    else:
        while learnrate > lr_limit:
            if k < 4:
                opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=18e-08)
            else:
                opt = SGD(lr=learnrate, momentum=0.9, nesterov=True)
                epochs = 120
            vae.compile(loss=None, optimizer=opt, metrics=["mse"])
            vae.fit(
                x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                callbacks=[checkpointer, history, logger],
            )
            vae.load_weights(f"{options.data}.weights.hdf5")
            learnrate /= 2
            k = k + 1

            vae.save_weights(f"{options.data}.weights.h5")

    latent_mean = encoder.predict(x_train)[0]
    latent_logvar = encoder.predict(x_train)[1]
    latent_var = np.exp(latent_logvar)
    latent_std = np.sqrt(latent_var)
    np.savetxt(f"{options.data}.latent_mean.csv", latent_mean)
    np.savetxt(f"{options.data}.latent_std.csv", latent_std)
    filename = f"{options.data}.latent_mean.csv"
    means_df = pd.read_csv(filename, sep=" ", header=None)
    mean = means_df.values
    filename = f"{options.data}.latent_std.csv"
    stds_df = pd.read_csv(filename, sep=" ", header=None)
    std = stds_df.values
    lat_dim = 20
    b = "e-6"
    samples_num = data.shape[0]
    z_samples = np.empty([samples_num, lat_dim])

    # sampling from the new prior with gamma=0.05

    l = 0
    for i in range(0, samples_num):
        for j in range(0, lat_dim):
            z_samples[l, j] = np.random.normal(
                mean[i % 100000, j], 0.05 + std[i % 100000, j]
            )
        l = l + 1
    new_events = decoder.predict(z_samples)
    for i in range(0, new_events.shape[1]):
        new_events[:, i] = new_events[:, i] * max[i]

    np.savetxt(f"{options.data}.gen_events.csv", new_events)
