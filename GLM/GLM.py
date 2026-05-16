from keras.regularizers import l2
from tensorflow.keras.layers import Activation, Add, Dense, Flatten, Input
from tensorflow.keras.models import Model


def GLM(filter_shape, history_shape, activation="softplus", l2_reg=1e-3):   #retina:1e-4 cortex:1e-3
    ncell = history_shape[-1]

    stimulus_input = Input(shape=filter_shape, name="stimulus_input")
    flat_stimulus = Flatten()(stimulus_input)
    stimulus_filter = Dense(
        ncell,
        kernel_regularizer=l2(l2_reg),
        name="stimulus_filter",
    )(flat_stimulus)

    history_input = Input(shape=history_shape, name="history_input")
    flat_history = Flatten()(history_input)
    history_filter = Dense(
        ncell,
        kernel_regularizer=l2(l2_reg),
        name="history_filter",
    )(flat_history)

    linear_predictor = Add(name="linear_predictor")([stimulus_filter, history_filter])
    output = Activation(activation, name="rate")(linear_predictor)
    return Model(inputs=[stimulus_input, history_input], outputs=output, name="GLM")

