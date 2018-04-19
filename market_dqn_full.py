import os
import numpy as np
import random
import copy
from market_env_full import MarketEnv
from market_model_builder import MarketDeepQLearningModelBuilder
from collections import deque
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from os import walk



class DeepQ:

    def __init__(self, env, gamma=0.85, model_file_name=None, test = False, random = False):
    # def __init__(self, env, current_discount=0.85, gamma=0.85, model_file_name=None, test = False):
        self.env = env
        # self.cur_discount = current_discount
        self.gamma = gamma
        self.model_filename = model_file_name
        self.memory = deque(maxlen=1000000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.005
        self.random = random

        self.model = MarketDeepQLearningModelBuilder().buildModel()
        self.fixed_model = MarketDeepQLearningModelBuilder().buildModel()

        # rmsprop = RMSprop(lr=0.0001)
        # self.model.compile(loss='mse', optimizer=rmsprop)

        if test:
            self.model.load_weights(self.model_filename)

    def epsilon_reset(self):
        self.epsilon = 1.0


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):

        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs)
        losses = []

        true_q = []
        train_states = []
        self.fixed_model.set_weights(self.model.get_weights())
        for i in batchs:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                predict_f = self.fixed_model.predict(next_state)[0]

                # predict_f = self.model.predict(next_state)[0]
                # target = self.cur_discount * reward + self.discount * np.amax(predict_f)
                target = reward + self.gamma * np.amax(predict_f)

            target_f = self.model.predict(state)

            target_f[0][action] = target
            

            train_states.append(state[0])
            true_q.append(target_f)
            # print(target_f)
            # exit()

            # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)
        train_states = np.array(train_states)
        true_q = np.array(true_q)
        history = self.model.fit(state, target_f, epochs=1, verbose=0)
        losses=history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return losses

    def act(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        # cur_baseAmount = info["baseAmount"]


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= self.epsilon:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] # returns action

    def act_predict_new(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        # cur_baseAmount = info["baseAmount"]


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= 0.18:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] # returns action


    def act_random(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)
        
        sample = self.env.action_space.sample()
        
        while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()
        return sample


    def act_predict(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        # cur_baseAmount = info["baseAmount"]


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        act_values = self.model.predict(state)[0]
        # print(act_values)

        act_index = np.argsort(act_values)[::-1]
        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1

        # print (act_index[flag])

        return act_index[flag] # returns action



    # def find_action_index(self, true_action):
    #     for i in range(len(self.env.actions)):
    #         if self.env.actions[i] == true_action:
    #             return i
    #     print("error!")

    def train(self, code_stocks, max_episode=50, verbose=0):

        history = open('./record/history.txt', 'a')

        # if os.path.exists("./record/train_loss.txt"):
        #     os.remove("./record/train_loss.txt")

        for e in range(max_episode):
            self.env._reset(code_stocks)
            state = self.env._render()
            # self.epsilon_reset()

            game_over = False
            reward_sum = 0

            holds = 0
            buys = 0
            sells = 0
            print ("----",self.env.targetCode,"----")
            info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": 0, \
                "current_asset":{'cash':self.env.startAssetValue, 'stock':0}}
            while not game_over:

                action = self.act(state, info)

                if self.env.actions[action] == 'Hold':
                    holds += 1
                elif self.env.actions[action] == 'Buy':
                    buys += 1
                elif self.env.actions[action] == 'Sell':
                    sells += 1
                next_state, reward, game_over, info = self.env._step(action)
                
                current_asset_value = info['current_asset_value']
                # print info
                self.remember(state, action, reward, next_state, game_over)

                state = copy.deepcopy(next_state)

                # reward_sum += reward*float(info["last_asset_value"])

                if game_over:
                    toPrint = '----episode----', e,'totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                    # toPrint = '----episode----', e,'   totalrewards:', round(reward_sum/float(10000), 4), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                    print (toPrint)
                    history.write(str(toPrint))
                    history.write('\n')
                    history.flush()

            # if e % 20 == 0 and e != 0:
                # pre_string = self.model_filename.split(".")[1] + '_' + str(int(e/200))
                # tmp_filename = "." + pre_string + ".h5"
                # self.model.save_weights(tmp_filename)
            self.model.save_weights(self.model_filename)
            print ('model weights saved')

            losses = self.replay(100)

            with open("./record/train_loss.txt","a") as file:
                file.write(str(losses)+'\n')

            print("Loss over 128 samples: " + str(losses)+ '\n')

        history.close()

    def predict(self,code_stocks):
        if os.path.exists('./record/test_history.txt'):
            os.remove('./record/test_history.txt')

        history = open('./record/test_history.txt', 'a')

        # self.env._reset()
        self.env._reset(code_stocks)
        state = self.env._render()

        game_over = False
        reward_sum = 0

        holds = 0
        buys = 0
        sells = 0

        if self.random == True:
            methods = 'random'
        else:
            methods = 'DQN'


        print ("----",self.env.targetCode,"----")
        info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": 0, \
                "current_asset":{'cash':self.env.startAssetValue, 'stock':0}}

        while not game_over:

            if self.random == True:
                action = self.act_random(state,info)
            else:
                action = self.act_predict_new(state,info)

            if self.env.actions[action] == 'Hold':
                holds += 1
            elif self.env.actions[action] == 'Buy':
                buys += 1
            elif self.env.actions[action] == 'Sell':
                sells += 1

            next_state, reward, game_over, info = self.env._step(action)
            
            current_asset_value = info['current_asset_value']

            state = copy.deepcopy(next_state)

            # reward_sum += reward

            if game_over:
                toPrint = '----episode----', methods,'totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                print (toPrint)
                history.write(str(toPrint))
                history.write('\n')
                history.flush()

        history.close()
        return round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3)

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files

def plot_train_loss(path):
    with open(path, "r") as file:
        losses = file.read().split('\n')
        losses = losses[:-1]
        losses = [float(each) for each in losses]

    plt.plot(losses)
    plt.title("training loss")
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.show()

def plot_train_reward(path):
    rewards = []
    with open(path, "r") as file:
        for eachline in file:
            reward = eachline.split(',')
            rewards.append(float(reward[3]))

    plt.plot(rewards)
    plt.title("Rewards")
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.show()

if __name__ == "__main__":
    
    files = exploreFolder('./semantic_data/noadd')
    # files.remove('.DS_Store')
   
    # s_and_p = ['ESS', 'MMM', 'COO', 'SPG', 'WHR', 'ROP', 'AMG', 'IBM', 'GS', 'RE', 'AVB', 'GWW', 'MCK', 'PXD', 'MHK', 'PSA']
    s_and_p = [x for x in files]
 
    s_and_p_test = sorted(s_and_p)

    if os.path.exists("./record/train_loss.txt"):
        os.remove("./record/train_loss.txt")

    if os.path.exists("./record/history.txt"):
        os.remove("./record/history.txt")
    
    env = MarketEnv(dir_path="./semantic_data/noadd/", target_codes=s_and_p, sudden_death_rate=0.3, finalIndex = 370) #370
    pg = DeepQ(env, gamma=0.80, model_file_name="./model/model_ml3_noadd.h5")
    
    for stock in s_and_p:
        # pg.epsilon_reset()
        pg.train(code_stocks = stock)


    reward_stock = []
    reward_stock_random = []


    env = MarketEnv(dir_path="./semantic_data/noadd/",target_codes=s_and_p_test, test = True, sudden_death_rate=0.3, finalIndex = 131) #131
    test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/model_ml3_noadd.h5", test = True)
    test_obj_random = DeepQ(env, gamma=0.80, model_file_name="./model/model_ml3_noadd.h5", test = True, random = True)

    for stock in s_and_p_test:
        # reward_stock.append(test_obj.predict(stock))

    # print(s_and_p_test)
    # print(reward_stock)
        reward_collect = []
        reward_collect_random = []

        for i in range(20):
            each_reward = test_obj.predict(stock) 
            reward_collect.append(each_reward)
            each_reward = test_obj_random.predict(stock)
            reward_collect_random.append(each_reward)

        reward_stock.append(np.mean(reward_collect))
        reward_stock_random.append(np.mean(reward_collect_random))

    sum_0 = 0
    sum_1 = 0
    size = len(s_and_p)
    for reward in reward_stock:
        sum_0+=reward
    for reward in reward_stock_random:
        sum_1+=reward

    print(s_and_p_test)
    print(reward_stock)

    print(sum_0)
    print(sum_1)
    print((sum_0-sum_1)/sum_1)

    plt.plot(s_and_p_test,reward_stock, c='tab:blue')
    plt.plot(s_and_p_test,reward_stock_random, c='tab:gray')
    # plt.plot(s_and_p_test,reward_collect)
    # plt.plot(s_and_p_test,reward_collect_random)
    plt.title("DQN vs Random")
    plt.xlabel("stock")
    plt.ylabel("return")
    plt.legend(['DQN', 'Random'], loc='upper left')
    plt.show()

    # env = MarketEnv(dir_path="./sample_data/", target_codes="training_sp", sudden_death_rate=0.3, finalIndex = 1997)

    # env = MarketEnv(dir_path="./test_data/", target_codes="test_AAPL", sudden_death_rate=0.3, finalIndex = 517)

    # pg = DeepQ(env, current_discount=0.66, gamma=0.80, model_file_name="./model/growing/train_model.h5")

    # all_rewards=[]

    # for i in range(24):
    #     pre_string = "./model/growing/train_model.h5".split(".")[1] + '_' + str(i+1)
    #     model_filename = "." + pre_string + ".h5"
    #     test_obj = DeepQ(env=env, current_discount=0.66, gamma=0.80, model_file_name=model_filename, test = True)
    	
    #     each_reward = test_obj.predict()
    #     all_rewards.append(each_reward)


    # pg.train()



####################		uncomment the following if test  		###################
   	# plt.plot(all_rewards)
    # plt.title("Test Rewards")
    # plt.xlabel("episodes")
    # plt.ylabel("rewards")
    # plt.show()



    # targetCodes = exploreFolder('sample_data')
    # test_targetCodes = exploreFolder('test_data')
    # targetCodes.remove('.DS_Store')


    plot_train_loss("./record/train_loss.txt")
    plot_train_reward("./record/history.txt")
    exit()



    # def replay(self, batch_size):

    #     batchs = min(batch_size, len(self.memory))
    #     batchs = np.random.choice(len(self.memory), batchs)
    #     losses = []
    #     for i in batchs:
    #         state, action, reward, next_state, done = self.memory[i]
    #         target = reward
    #         if not done:
    #             predict_f = self.model.predict(next_state)[0]
    #             # target = self.cur_discount * reward + self.discount * np.amax(predict_f)
    #             target = reward + self.gamma * np.amax(predict_f)


    #         target_f = self.model.predict(state)


    #         target_f[0][action] = target

    #         # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)

    #         history = self.model.fit(state, target_f, epochs=1, verbose=0)
    #         losses.append(history.history['loss'][0])
            

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    #     return losses




    # s_and_p = sorted(random.sample(s_and_p_initial, 30))
    # print(s_and_p)
    # ['ADI', 'AJG', 'APD', 'CHD', 'CVX', 'DLR', 'DVA', 'EQT', 'ETN', 'HBI', 'HES', 'INTU', 'IT', 'KMI', 'L', 'MAR', 'MET', 'MMM', 
    # 'NAVI', 'NOC', 'NSC', 'PLD', 'SLG', 'SPGI', 'SRCL', 'TJX', 'TMO', 'TSCO', 'TWX', 'UAA']
    # s_and_p = ['ADI', 'AJG', 'APD', 'CVX', 'DLR', 'DVA', 'ETN', 
    # 'HES', 'INTU', 'IT','L', 'MAR', 'MET', 'MMM', 'NOC', 'NSC', 'PLD', 'SPGI', 'TJX', 'TMO']

    # s_and_p = ['TMO']


    # test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/stock_model/model_"+stock+".h5", test = True)
    # test_obj_random = DeepQ(env, gamma=0.80, model_file_name="./model/stock_model/model_"+stock+".h5", test = True, random = True)











