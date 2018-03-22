from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class MarketDeepQLearningModelBuilder():
   
    def __init__(self, state_size = 9, action_size = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001

    def buildModel(self):
        model = Sequential()
        model.add(Dense(48, input_shape = (self.state_size,), activation='relu'))
        model.add(Dense(96, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(96, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        print (model.summary())

        return model



# from keras.models import Model
# from keras.layers import merge, Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
# from keras.layers.merge import concatenate
# from keras.layers.advanced_activations import LeakyReLU
    # def buildModel(self):


        # merges = []
        # inputs = []

        # """B0"""
        # B0 = Input(shape=(2, ))
        # b = Dense(7, activation="sigmoid")(B0)

        # inputs.append(B0)
        # merges.append(b)

        # """S0"""
        # S0 = Input(shape=(7, 60, 1, ))

        # # s = Flatten()(S0)
        # s = Conv2D(filters=1024, kernel_size=(1, 60), strides=1, padding='valid', data_format='channels_last')(S0)
        # s = LeakyReLU(0.001)(s)
        # s = Flatten()(s)
        # s = Dense(90)(s)
        # s = LeakyReLU(0.001)(s)

        # inputs.append(S0)
        # merges.append(s)


        # m = concatenate(merges, axis=1)
        # m = Dense(90)(m)
        # m = LeakyReLU(0.001)(m)
        # m = Dense(30)(m)
        # m = LeakyReLU(0.001)(m)
                # V = Dense(3, activation='linear')(m)

        # model = Model(inputs=inputs, outputs=V)





