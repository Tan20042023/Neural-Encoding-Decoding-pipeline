from tf_keras.layers import Activation, Conv2D, Dense, Flatten, GaussianNoise
from tf_keras.models import Model
from tf_keras.regularizers import l2


def CNN(inputs, n_out):
    sigma = 0.05
    x = Conv2D(16, 15, padding="same", data_format="channels_last")(inputs)
    x = Activation("relu")(GaussianNoise(sigma)(x))
    x = Conv2D(8, 9, padding="same", data_format="channels_last")(x)
    x = Activation("relu")(GaussianNoise(sigma)(x))
    x = Flatten()(x)
    x = Dense(n_out, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    outputs = Activation("softplus")(x)
    return Model(inputs, outputs, name="CNN")

