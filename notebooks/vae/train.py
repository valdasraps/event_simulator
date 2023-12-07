from optparse import OptionParser

import numpy as np
import pandas as pd
from keras import backend as K
from keras.src.callbacks import History, ModelCheckpoint
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.losses import mse
from keras.src.optimizers import SGD, Adam

from notebooks.vae.model import get_vae_decoder_encoder

if __name__ == "__main__":
    parser = OptionParser(usage="%prog -d %data_file")
    parser.add_option(
        "-d", "--data", dest="data", help="Training data file", metavar="data"
    )

    (options, args) = parser.parse_args()

    if options.data is None:
        parser.error("Data file not provided")
        exit(1)

    ttb_df = pd.read_csv(options.data, sep=" ", header=None)
    data = ttb_df.values
    data = data[:, 0:26]

    train_size = 100000

    x_train = data[:train_size]
    x_test = data[100000:200000]
    image_size = x_train.shape[1]
    input_dim = image_size

    x_train = np.reshape(x_train, [-1, input_dim]).astype("float32")
    x_test = np.reshape(x_test, [-1, input_dim]).astype("float32")

    epochs = 240
    batch_size = 1024

    def vae_loss(
        x: KerasTensor,
        x_decoded_mean: KerasTensor,
        z_log_var: KerasTensor,
        z_mean: KerasTensor,
    ) -> KerasTensor:
        mse_loss = mse(x, x_decoded_mean)
        kl_loss = z_log_var + 1 - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        beta = 10 ** (-6)
        loss = K.mean((1 - beta) * mse_loss + beta * kl_loss)
        return loss

    vae, decoder, encoder = get_vae_decoder_encoder(input_dim=input_dim)

    learnrate = 0.001
    iterations = 7
    lr_limit = 0.001 / (2**iterations)
    history = History()
    k = 0
    checkpointer = ModelCheckpoint(
        filepath="ttbar_20d_e-6.hdf5", verbose=1, save_best_only=True
    )
    opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    vae.add_loss(
        vae_loss(
            x=vae.input,
            x_decoded_mean=vae.output,
            z_log_var=decoder.outputs[1],
            z_mean=decoder.outputs[0],
        )
    )
    vae.compile(optimizer=opt, loss=None)

    vae.summary()

    k = 0
    while learnrate > lr_limit:
        if k < 4:
            opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
        vae.load_weights("ttbar_20d_e-6.hdf5")
        learnrate /= 2
        k = k + 1

        vae.save_weights("ttbar_20d_e-6.h5")
