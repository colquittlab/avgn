import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import tqdm

from avgn.tensorflow.VAE2 import plot_reconstruction, log_normal_pdf
from avgn.utils.paths import ensure_dir
from IPython.display import clear_output

import avgn.tensorflow.data as tfdata
import keras

ds = tfp.distributions

@keras.saving.register_keras_serializable(package="Custom")
class VAE(tf.keras.Model):
    """a basic vae class for tensorflow
    Extends:
        tf.keras.Model
    """
    def __init__(self, beta=1.0, enc=None, dec=None, N_Z = 32, DIMS = (128,128,1), batch_size = None, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)
        self.beta = beta
        self.N_Z = N_Z
        self.DIMS = DIMS

        # Ensure enc and dec are valid Keras Sequential models
        self.enc = self._initialize_model(enc, self.default_encoder())
        self.dec = self._initialize_model(dec, self.default_decoder())

        # Batch size handling
        self.batch_size = batch_size if batch_size is not None else None

    def _initialize_model(self, model, default_layers):
        """Helper method to initialize enc/dec as Sequential models."""
        if isinstance(model, tf.keras.Sequential):
            return model
        elif isinstance(model, list):
            return tf.keras.Sequential(model)
        elif model is None:
            return tf.keras.Sequential(default_layers)
        else:
            raise ValueError("Model must be a Sequential model, list of layers, or None.")
        
    def default_encoder(self):
        return [        tf.keras.layers.InputLayer(shape=self.DIMS),
                        tf.keras.layers.Conv2D(
                            filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2D(
                            filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2D(
                            filters=128, kernel_size=3, strides=(2, 2), activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2D(
                            filters=256, kernel_size=3, strides=(2, 2), activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2D(
                            filters=256, kernel_size=3, strides=(2, 2), activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(units=self.N_Z*2),
        ]

    def default_decoder(self):
        return [        tf.keras.layers.Dense(units=4 * 4 * 256, activation=tf.nn.leaky_relu),
                        tf.keras.layers.Reshape(target_shape=(4, 4, 256)),
                        tf.keras.layers.Conv2DTranspose(
                            filters=256, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2DTranspose(
                            filters=256, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2DTranspose(
                            filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2DTranspose(
                            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2DTranspose(
                            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu
                        ),
                        tf.keras.layers.Conv2DTranspose(
                            filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
               ]

    def get_config(self):
        """Returns the configuration of the model for serialization."""
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "enc": keras.saving.serialize_keras_object(self.enc),
            "dec": keras.saving.serialize_keras_object(self.dec),
            "batch_size": self.batch_size,
            "input_shape": (self.batch_size,) + self.enc.input_shape[1:],
            "N_Z": self.N_Z,
            "DIMS": self.DIMS
                      })
        return config

    def get_build_config(self):
        return {
            "enc": keras.saving.serialize_keras_object(self.enc),
            "dec": keras.saving.serialize_keras_object(self.dec),
        }
    
    @classmethod
    def from_config(cls, config):
        """Creates an instance of the model from a configuration dictionary."""
        enc = keras.saving.deserialize_keras_object(config.pop("enc"))
        dec = keras.saving.deserialize_keras_object(config.pop("dec"))
        batch_size = config.pop("batch_size", None)
        N_Z = config.pop("N_Z")
        DIMS = config.pop("DIMS")
        
        return cls(enc=enc, dec=dec, batch_size=batch_size, N_Z=N_Z, **config)

    @classmethod
    def build_from_config(cls, config):
        enc_config = config.get("enc")
        dec_config = config.get("dec")
        beta_config = config.get("beta", 1.0)  # Default beta to 1.0 if missing
        N_Z_config = config.get("N_Z"),
        DIMS_config = config.get("DIMS")
    
        if enc_config is None or dec_config is None:
            raise KeyError("Missing 'encoder' or 'decoder' in the config dictionary.")
    
        enc = keras.saving.deserialize_keras_object(enc_config)
        dec = keras.saving.deserialize_keras_object(dec_config)
        
        return cls(beta=beta_config, enc=enc, dec=dec)

    def encode(self, x):
        mean, logvar = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def reconstruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu, apply_sigmoid=True)

    def decode(self, z, apply_sigmoid=False):
        logits = self.dec(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        # reconstruction
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        recon_loss = -tf.reduce_mean(logpx_z)
        # kl
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar) * self.beta
        latent_loss = -tf.reduce_mean(logpz - logqz_x)

        return recon_loss, latent_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train_net(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            latent_dim = int(self.enc.variables[-1].shape[0] / 2)
            eps = tf.random.normal(shape=(100, latent_dim))
        return self.decode(eps, apply_sigmoid=True)

def plot_reconstruction(model, example_data, N_Z, nex=8, zm=2):

    example_data_reconstructed = model.reconstruct(example_data)
    samples = model.decode(tf.random.normal(shape=(len(example_data), N_Z)))
    fig, axs = plt.subplots(ncols=nex, nrows=3, figsize=(zm * nex, zm * 3))
    for axi, (dat, lab) in enumerate(
        zip(
            [example_data, example_data_reconstructed, samples],
            ["data", "data recon", "samples"],
        )
    ):
        for ex in range(nex):
            axs[axi, ex].matshow(
                dat.numpy()[ex].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
            )
            axs[axi, ex].axes.get_xaxis().set_ticks([])
            axs[axi, ex].axes.get_yaxis().set_ticks([])
        axs[axi, 0].set_ylabel(lab)

    plt.show()

def save_dataset_to_tfrecord(dataset, 
                             filename=None                         
):
    
    """Saves a TensorFlow dataset to a TFRecord file.
    Args:
        dataset (tf.data.Dataset): The dataset to save.
        filename (str or Path): The path to the TFRecord file.
    Raises:
        ValueError: If `filename` is None or not a valid string or Path.
    """

    if filename is None:
        raise ValueError("Please provide a valid filename for the TFRecord file.")
    
    filename = str(filename)  # Ensure string path
    
    with tf.io.TFRecordWriter(filename) as writer:
        for example in dataset:
            spectrogram, idx, indv, label = example
    
            serialized = tfdata.serialize_example(
                 {
                    "spectrogram": {
                        "data": spectrogram.numpy().tobytes(),       # bytes
                        "_type": tfdata._bytes_feature
                    },
                    "idx": {
                        "data": int(idx.numpy()),                  # int64
                        "_type": tfdata._int64_feature
                    },
                    "indv": {
                        "data": indv.numpy(),                     # already bytes
                        "_type": tfdata._bytes_feature
                    },
                    "label": {
                        "data": label.numpy(),                  # already bytes
                        "_type": tfdata._bytes_feature
                    },
                }
                )

            writer.write(serialized)

    return

def split_dataset(dataset, 
                  TOT_DATA=100000, 
                  TEST_SIZE=10000,
                  TRAIN_BUF=1000, 
                  HOLDOUT_SIZE=10000, 
                  HOLDOUT_BUF = 1000, 
                  TEST_BUF=1000, 
                  save_dataset=False, 
                  save_loc=None, 
                  hold_out=True, 
                  shuffle=True
):
    """Splits a dataset into train, test, and holdout sets.
    Aergs:
        dataset (tf.data.Dataset): The dataset to split.
        TOT_DATA (int): Total number of samples in the dataset.
        TEST_SIZE (int): Number of samples for the test set.
        TRAIN_BUF (int): Buffer size for shuffling the training set.
        HOLDOUT_SIZE (int): Number of samples for the holdout set.
        HOLDOUT_BUF (int): Buffer size for shuffling the holdout set.
        TEST_BUF (int): Buffer size for shuffling the test set.
        save_dataset (bool): Whether to save the datasets to disk.
        save_loc (str or Path): Location to save the datasets.
        hold_out (bool): Whether to create a holdout set.
        shuffle (bool): Whether to shuffle the datasets.
    Returns:
        tuple: A tuple containing the train, test, and optionally holdout datasets.
    Raises:
        ValueError: If `save_loc` is not provided when `save_dataset` is True
    """

    if shuffle:
        TRAIN_SIZE = TOT_DATA - TEST_SIZE - HOLDOUT_SIZE
        
        test_dataset = dataset.take(TEST_SIZE).shuffle(TEST_BUF)
        
        if hold_out:
            TRAIN_SIZE = TOT_DATA - TEST_SIZE - HOLDOUT_SIZE
            
            test_dataset = dataset.take(TEST_SIZE).shuffle(TEST_BUF)
            holdout_dataset = dataset.skip(TEST_SIZE).take(HOLDOUT_SIZE).shuffle(HOLDOUT_BUF)
            train_dataset = dataset.skip(TEST_SIZE + HOLDOUT_SIZE).take(TRAIN_SIZE).shuffle(TRAIN_BUF)
            
        else:
           
            TRAIN_SIZE = TOT_DATA - TEST_SIZE 
            
            test_dataset = dataset.take(TEST_SIZE).shuffle(TEST_BUF)
            train_dataset = dataset.skip(TEST_SIZE).take(TRAIN_SIZE).shuffle(TRAIN_BUF)
            
    else:
        print('No shuffle, only returning shuffled-test and nonshuffled-train dataset')
        
        hold_out = False
        
        TRAIN_SIZE = TOT_DATA 
        
        train_dataset = dataset.take(TRAIN_SIZE)
        test_dataset = train_dataset.shuffle(TEST_BUF).take(TEST_SIZE)

    if save_dataset:
        if save_loc is None:
            raise ValueError("Please provide a valid save location.")
        ensure_dir(save_loc)
        print(f"Saving dataset to: {save_loc}.tfrecord")
        
        save_dataset_to_tfrecord(test_dataset, save_loc / 'test.tfrecord')
        
        if hold_out:
            save_dataset_to_tfrecord(holdout_dataset, save_loc / 'holdout.tfrecord')
            
        save_dataset_to_tfrecord(train_dataset, save_loc / 'train.tfrecord')
        
    if hold_out:
        return train_dataset, test_dataset, holdout_dataset
    else:
        return train_dataset, test_dataset

def plot_losses(losses):
    """Plots the losses from a DataFrame.
    Args:
        losses (pd.DataFrame): DataFrame containing the losses with columns 'recon_loss' and 'latent_loss'.
    """

    cols = list(losses.columns)
    fig, axs = plt.subplots(ncols = len(cols), figsize= (len(cols)*4, 4))
    for ci, col in enumerate(cols):
        if len(cols) == 1:
            ax = axs
        else:
            ax = axs.flatten()[ci]
        ax.loglog(losses[col].values)
        ax.set_title(col)
    plt.show()

def train_VAE(model = None, 
              train_dataset=None, 
              test_dataset=None, 
              BATCH_SIZE=128, 
              epoch=0, 
              n_epochs = 100, 
              DIMS = (128, 128, 1), 
              N_TRAIN_BATCHES = 100, 
              N_TEST_BATCHES = 15, 
              N_Z=32
):
    """Trains a VAE model on the provided datasets.
    Args:
        model (VAE): The VAE model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        test_dataset (tf.data.Dataset): The testing dataset.
        BATCH_SIZE (int): The batch size for training.
        epoch (int): The starting epoch for training.
        n_epochs (int): The total number of epochs to train.
        DIMS (tuple): The dimensions of the input data.
        N_TRAIN_BATCHES (int): Number of training batches per epoch.
        N_TEST_BATCHES (int): Number of testing batches per epoch.
        N_Z (int): Dimensionality of the latent space.
    Returns:
        tuple: A tuple containing the trained model, test dataset, train dataset, and losses DataFrame
    Raises:
        ValueError: If any of the required datasets or model is not provided.
    """


    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    example_data = next(iter(test_dataset))
    example_data = (tf.cast(tf.reshape(example_data[0], [BATCH_SIZE] + list(DIMS)), tf.float32) / 255 )
    
    if train_dataset is None:
        raise ValueError("Enter train dataset for training")

    if test_dataset is None:
        raise ValueError("Enter test dataset for training")
    
    if model is None:
        raise ValueError("Please provide a valid model for training.")

    # a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns = ['recon_loss', 'latent_loss'])

    for epoch in range(epoch, n_epochs):
        # train
        for batch, train_x in tqdm(zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES):
            x = tf.cast(tf.reshape(train_x[0], [BATCH_SIZE] + list(DIMS)), tf.float32) / 255
            model.train_net(x)
            
        # test on holdout
        loss = []
        
        for batch, test_x in tqdm(zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES):
            x = tf.cast(tf.reshape(test_x[0], [BATCH_SIZE] + list(DIMS)), tf.float32) / 255
            loss.append(model.compute_loss(x))
        
        losses.loc[len(losses)] = np.mean(loss, axis=0)

        # plot results
        clear_output(wait=True)
        print("Epoch: {} | recon_loss: {} | latent_loss: {}".format(epoch, losses.recon_loss.values[-1], losses.latent_loss.values[-1]))
        
        plot_reconstruction(model, example_data, N_Z=N_Z)
        plot_losses(losses)
        
        fig, ax = plt.subplots()
        z = tf.split(model.enc(example_data), num_or_size_splits=2, axis=1)[0].numpy()
        ax.hist(z.flatten(), bins = 50)
        plt.show()
        
    return model, test_dataset, train_dataset, losses

def save_model(model=None, 
               save_loc=None
):
    """Saves a Keras model to a specified location.
    Args:
        model (tf.keras.Model): The model to save.
        save_loc (str or Path): The location to save the model.
    Raises:
        ValueError: If `model` is None or `save_loc` is None.
    """
    # Check if model and save_loc are provided
    if model is None:
        raise ValueError("Please provide a valid model to save.")
    if save_loc is None:
        raise ValueError("Enter save location")

    ensure_dir(save_loc)
    
    # Replace input_shape with the correct input shape for your model
    model.build(input_shape=model.get_config()['input_shape'])  # 128 is for the batch dimension
    
    # Save the entire model to a keras file.
    model.save(save_loc)

    print(f'model saved to: {save_loc}')
    
    return
