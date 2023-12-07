from optparse import OptionParser

import numpy as np
import pandas as pd
from keras import backend as K
from keras.src.callbacks import History, ModelCheckpoint
from keras.src.engine.base_layer import Layer
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.losses import mse
from keras.src.optimizers import SGD  # , Adam
from model import get_vae_encoder_decoder
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
    data = ttb_df.values[:, 0:26].copy()

    max = np.empty(26)
    for i in range(0, data.shape[1]):
        max[i] = np.max(np.abs(data[:, i]))
        if np.abs(max[i]) > 0:
            data[:, i] = data[:, i] / max[i]
        else:
            pass

    train_size = 100000

    x_train = data[:train_size]
    x_test = data[train_size : train_size * 2]
    image_size = x_train.shape[1]
    input_dim = image_size

    x_train = np.reshape(x_train, [-1, input_dim]).astype("float32")
    x_test = np.reshape(x_test, [-1, input_dim]).astype("float32")

    epochs = 240
    batch_size = 1024

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

    vae, encoder, decoder = get_vae_encoder_decoder(input_dim=input_dim)

    learnrate = 0.001
    iterations = 7
    lr_limit = 0.001 / (2**iterations)
    history = History()
    k = 0
    checkpointer = ModelCheckpoint(
        filepath=f"{options.data}.weights.hdf5", verbose=1, save_best_only=True
    )
    opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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
                callbacks=[checkpointer, history],
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
z_samples = np.empty([1200000, lat_dim])
l = 0

# sampling from the new prior with gamma=0.05

l = 0
for i in range(0, 1200000):
    for j in range(0, lat_dim):
        z_samples[l, j] = np.random.normal(
            mean[i % 100000, j], 0.05 + std[i % 100000, j]
        )
    l = l + 1
new_events = decoder.predict(z_samples)
for i in range(0, new_events.shape[1]):
    new_events[:, i] = new_events[:, i] * max[i]

np.savetxt(f"{options.data}.gen_events.csv", new_events)
