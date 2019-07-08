from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.models import Model


def build_model(args, cfg={}):
    observation_input = Input(shape=(5, 658))

    lstm1 = LSTM(512)(observation_input)
    h1 = Dense(512, activation=Activation('relu'))(lstm1)
    d1 = Dropout(0.25)(h1)
    h2 = Dense(512, activation=Activation('relu'))(d1)

    action_output = Dense(20, activation=Activation('softmax'))(h2)

    return Model(inputs=observation_input, outputs=action_output)
