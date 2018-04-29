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
        model.add(Dense(96, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        print (model.summary())

        return model






