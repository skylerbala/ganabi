from keras.layers import Input, Dense, Activation
from keras.models import Model


def build_model(args, cfg={}):
    observation_input = Input(shape=(658,))

    h1 = Dense(16, activation=Activation('relu'))(observation_input)
    h2 = Dense(16, activation=Activation('relu'))(h1)
    h3 = Dense(16, activation=Activation('relu'))(h2)

    action_output = Dense(20, activation=Activation('relu'))(h3)

    return Model(inputs=observation_input, outputs=action_output)
