from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf
from cosine_scheduler import CosineScheduler
from keras import backend as K
from keras.src.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
from keras.src.engine.base_layer import Layer
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.losses import mse
from keras.src.optimizers import Adagrad
from model import get_vae_encoder_decoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam

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

    epochs = 2400
    batch_size = 1024

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
        # inv_m_loss = tf.square(inv_mass(x) - inv_mass(x_decoded_mean))
        beta = 10 ** (-6)
        # alpha = 10 ** (-6)
        loss = K.mean((1 - beta) * mse_loss + beta * kl_loss)  # + alpha * inv_m_loss)
        return loss

    vae, encoder, decoder = get_vae_encoder_decoder(input_dim=input_dim)

    learning_rate = 0.001
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
    )
    lr_scheduler = LearningRateScheduler(
        CosineScheduler(100, warmup_steps=0, base_lr=0.01, final_lr=1e-6), verbose=1
    )

    # lr_scheduler = LearningRateScheduler(
    #     lambda epoch: learning_rate * epoch ** (-0.4) if epoch > 0 else learning_rate,
    #     verbose=1,
    # )
    # checkpointer = ModelCheckpoint(
    #     filepath=f"{options.data}.weights.hdf5", verbose=1, save_best_only=True
    # )
    logger = CSVLogger(f"{options.data}.log", separator=",", append=True)
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    vae.add_loss(
        vae_loss(
            x=vae.input,
            x_decoded_mean=vae.output,
            z_log_var=encoder.output[1],
            z_mean=encoder.output[0],
        )
    )
    vae.compile(optimizer=opt, loss=None)

    vae.summary()

    k = 0
    if options.weights:
        vae.load_weights(options.weights)
    else:
        vae.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test),
            callbacks=[early_stop, logger, lr_scheduler],
        )
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
