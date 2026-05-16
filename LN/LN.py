from tf_keras.layers import Activation, Dense, Flatten
from tf_keras.models import Model
from tf_keras.regularizers import l2


def LN(inputs, cell_num, activation="softplus", l2_reg=1e-3): #retina:1e-4 cortex:1e-3
    if activation not in ("softplus", "sigmoid", "relu", "exp"):
        raise ValueError(
            f"Unsupported activation '{activation}'. "
            "Choose from: softplus, sigmoid, relu, exp."
        )

    x = Flatten()(inputs)
    x = Dense(cell_num, kernel_regularizer=l2(l2_reg))(x)
    outputs = Activation(activation)(x)
    return Model(inputs=inputs, outputs=outputs, name=f"LN-{activation}")

