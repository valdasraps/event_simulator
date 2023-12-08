import os
from optparse import OptionParser

import numpy as np
import pandas as pd
from keras import Model
from keras import backend as K
from keras.src.callbacks import CSVLogger, History, ModelCheckpoint
from keras.src.engine.base_layer import Layer
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.losses import mse
from keras.src.optimizers import SGD  # , Adam
from model import get_vae_edis_encoder_decoder_disc
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam


def rm_file(file_name: str) -> str:
    try:
        os.remove(file_name)
    except OSError:
        pass
    return file_name


def set_trainable(model: Model, flag: bool):
    for layer in model.layers:
        layer.trainable = flag
    model.trainable = flag


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

    vae, edis, encoder, decoder, disc = get_vae_edis_encoder_decoder_disc(
        input_dim=input_dim, latent_dim=latent_dim
    )

    learnrate = 0.001
    iterations = 7
    lr_limit = 0.001 / (2**iterations)
    history = History()

    vae_checkpointer = ModelCheckpoint(
        filepath=rm_file(f"{options.data}.vae.weights.hdf5"),
        verbose=1,
        save_best_only=True,
    )
    vae_logger = CSVLogger(
        rm_file(f"{options.data}.vae.log"), separator=",", append=True
    )
    vae_opt = Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    vae.add_loss(
        vae_loss(
            x=vae.input,
            x_decoded_mean=vae.output,
            z_log_var=encoder.output[1],
            z_mean=encoder.output[0],
        )
    )
    vae.compile(optimizer=vae_opt, loss=None)
    vae.summary()

    disc_checkpointer = ModelCheckpoint(
        filepath=rm_file(f"{options.data}.disc.weights.hdf5"),
        verbose=1,
        save_best_only=True,
    )
    disc_logger = CSVLogger(
        filename=rm_file(f"{options.data}.disc.log"), separator=",", append=True
    )
    disc.compile(optimizer=Adam(learning_rate=learnrate), loss="binary_crossentropy")

    edis_checkpointer = ModelCheckpoint(
        filepath=rm_file(f"{options.data}.edis.weights.hdf5"),
        verbose=1,
        save_best_only=True,
    )
    edis_logger = CSVLogger(
        rm_file(f"{options.data}.edis.log"), separator=",", append=True
    )
    edis.compile(optimizer=Adam(learning_rate=learnrate), loss="binary_crossentropy")

    k = 0
    if options.weights:
        vae.load_weights(options.weights)
    else:
        while learnrate > lr_limit:
            set_trainable(vae, True)
            set_trainable(encoder, True)
            set_trainable(decoder, True)

            if k < 4:
                opt = Adam(
                    learning_rate=learnrate, beta_1=0.9, beta_2=0.999, epsilon=18e-08
                )
            else:
                opt = SGD(learning_rate=learnrate, momentum=0.9, nesterov=True)
                epochs = epochs // 2

            vae.compile(loss=None, optimizer=opt, metrics=["mse"])
            vae.fit(
                x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                callbacks=[vae_checkpointer, history, vae_logger],
            )
            vae.load_weights(f"{options.data}.vae.weights.hdf5")

            set_trainable(disc, True)
            disc_train_pred = encoder.predict(x_train)[2]
            disc_fake_pred = np.random.standard_normal((x_train.shape[0], latent_dim))
            disc_train_x = np.concatenate([disc_train_pred, disc_fake_pred])
            disc_train_y = np.concatenate(
                [np.zeros(x_train.shape[0]), np.ones(x_train.shape[0])]
            )
            disc_test_pred = encoder.predict(x_test)[2]
            disc_fake_pred = np.random.standard_normal((x_test.shape[0], latent_dim))
            disc_test_x = np.concatenate([disc_test_pred, disc_fake_pred])
            disc_test_y = np.concatenate(
                [np.zeros(x_test.shape[0]), np.ones(x_test.shape[0])]
            )

            disc.compile(
                optimizer=Adam(learning_rate=learnrate), loss="binary_crossentropy"
            )
            disc.fit(
                disc_train_x,
                disc_train_y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(disc_test_x, disc_test_y),
                callbacks=[disc_checkpointer, history, disc_logger],
            )
            disc.load_weights(f"{options.data}.disc.weights.hdf5")

            set_trainable(edis, True)
            set_trainable(encoder, True)
            set_trainable(disc, False)
            edis.compile(
                optimizer=Adam(learning_rate=learnrate), loss="binary_crossentropy"
            )
            edis.fit(
                x_train,
                np.ones(x_train.shape[0]),
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, np.ones(x_test.shape[0])),
                callbacks=[edis_checkpointer, history, edis_logger],
            )
            edis.load_weights(f"{options.data}.edis.weights.hdf5")

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
