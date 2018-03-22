import os
import numpy as np
import random
import copy
from market_env import MarketEnv
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
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.  # exploration rate
        self.epsilon_decay = .98
        self.epsilon_min = 0.005
        self.random = random

        self.model = MarketDeepQLearningModelBuilder().buildModel()
        self.fixed_model = MarketDeepQLearningModelBuilder().buildModel()

        # rmsprop = RMSprop(lr=0.0001)
        # self.model.compile(loss='mse', optimizer=rmsprop)

        if test:
            self.model.load_weights(self.model_filename)


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
            true_q.append(target_f[0])

            # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)
        train_states = np.array(train_states)
        true_q = np.array(true_q)
        history = self.model.fit(train_states, true_q, epochs=1, verbose=0)
        losses=history.history['loss'][0]
        return losses

    def act(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]


        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general
        
        if np.random.rand() <= self.epsilon:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_buy and sample == 0) or \
            (cur_stock < cur_baseAmount and sample == 1) or (cur_cash < trading_fee_sell and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_buy and act_index[flag] == 0) or \
            (cur_stock < cur_baseAmount and act_index[flag]== 1) or \
            (cur_cash < trading_fee_sell and act_index[flag] == 1):
            flag+=1


        return act_index[flag] # returns action

    def act_random(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]

        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general
        sample = self.env.action_space.sample()
        
        while (cur_cash < trading_fee_buy and sample == 0) or \
        (cur_stock < cur_baseAmount and sample == 1) or (cur_cash < trading_fee_sell and sample == 1):
            sample = self.env.action_space.sample()
        return sample


    def act_predict(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]


        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general

        act_values = self.model.predict(state)[0]
        # print(act_values)

        act_index = np.argsort(act_values)[::-1]
        flag = 0

        while (cur_cash < trading_fee_buy and act_index[flag] == 0) or \
            (cur_stock < cur_baseAmount and act_index[flag]== 1) or \
            (cur_cash < trading_fee_sell and act_index[flag] == 1):
            flag+=1

        # print (act_index[flag])

        return act_index[flag] # returns action



    # def find_action_index(self, true_action):
    #     for i in range(len(self.env.actions)):
    #         if self.env.actions[i] == true_action:
    #             return i
    #     print("error!")

    def train(self, max_episode=900, verbose=0):

        history = open('./record/history.txt', 'w')

        if os.path.exists("./record/train_loss.txt"):
            os.remove("./record/train_loss.txt")

        for e in range(max_episode):
            self.env._reset()
            state = self.env._render()

            game_over = False
            reward_sum = 0

            holds = 0
            buys = 0
            sells = 0
            print ("----",self.env.code,"----")
            info = { "baseAmount": 2,"trading_feeRate": 0.001,"current_trading_price": 0, \
                "current_asset":{'cash':200, 'stock':0}}
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
                    toPrint = '----episode----', e,'totalgains:', round((current_asset_value-200)/200,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                    # toPrint = '----episode----', e,'   totalrewards:', round(reward_sum/float(10000), 4), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                    print (toPrint)
                    history.write(str(toPrint))
                    history.write('\n')
                    history.flush()

            if e % 20 == 0 and e != 0:
                # pre_string = self.model_filename.split(".")[1] + '_' + str(int(e/200))
                # tmp_filename = "." + pre_string + ".h5"
                # self.model.save_weights(tmp_filename)
                self.model.save_weights(self.model_filename)
                print ('model weights saved')

            losses = self.replay(128)

            with open("./record/train_loss.txt","a") as file:
                file.write(str(losses)+'\n')

            print("Loss over 128 samples: " + str(losses)+ '\n')

        history.close()

    def predict(self):
        if os.path.exists('./record/test_history.txt'):
            os.remove('./record/test_history.txt')

        history = open('./record/test_history.txt', 'a')

        self.env._reset()
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


        print ("----",self.env.code,"----")
        info = { "baseAmount": 2,"trading_feeRate": 0.001,"current_trading_price": 0, \
                "current_asset":{'cash':200, 'stock':0}}

        while not game_over:

            if self.random == True:
                action = self.act_random(state,info)
            else:
                action = self.act_predict(state,info)

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
                toPrint = '----episode----', methods,'totalgains:', round((current_asset_value-200)/200,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                print (toPrint)
                history.write(str(toPrint))
                history.write('\n')
                history.flush()

        history.close()
        return round((current_asset_value-200)/200,3)

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

    # s_and_p = ['ADM','ARNC','BLL','FITB','CBOE','DPS','PPG','KMX','LB','MMC','SCHW','TGT','UDR','VFC','DIS','WM','XYL','YUM','ZBH','ZTS']
    # s_and_p = ['ADM', 'ARNC', 'BLL', 'CBOE', 'DIS', 'DPS', 'FITB', 'KMX', 'LB','MMC', 'PPG', 'SCHW', 'TGT', 'UDR', 'VFC', 'WM', 'XYL', 'YUM','ZBH', 'ZTS']
    # s_and_p = ['DIS', 'DPS', 'FITB', 'KMX','LB','MMC', 'PPG', 'SCHW', 'TGT']

    s_and_p_initial = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
        'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
        'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
        'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV',
        'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC',
        'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BK',
        'BAX','BBT','BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX',
        'BHF','BMY','AVGO','BF.B','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
        'KMX','CCL','CAT','CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW',
        'CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG',
        'CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP',
        'ED','STZ','COO','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI',
        'DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
        'DG','DLTR','D','DOV','DWDP','DPS','DTE','DRE','DUK','DXC','ETFC','EMN','ETN',
        'EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR',
        'ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST',
        'FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV',
        'FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD',
        'GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC',
        'HSY','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII',
        'IDXX','INFO','ITW','ILMN','IR','INTC','ICE','IBM','INCY','IP','IPG','IFF','INTU',
        'ISRG','IVZ','IQV','IRM','JEC','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY',
        'KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK',
        'LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM',
        'MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU',
        'MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ',
        'NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI',
        'NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE',
        'ORCL','PCAR','PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE',
        'PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR',
        'PLD','PRU','PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RRC','RJF','RTN','O',
        'RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SBAC',
        'SCG','SLB','SNI','STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV',
        'SPGI','SWK','SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TPR',
        'TGT','TEL','FTI','TXN','TXT','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRV',
        'TRIP','FOXA','FOX','TSN','UDR','ULTA','USB','UAA','UA','UNP','UAL','UNH','UPS','URI',
        'UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO',
        'VMC','WMT','WBA','DIS','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WMB',
        'WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']

    # s_and_p = sorted(random.sample(s_and_p_initial, 30))
    # print(s_and_p)
    # ['ADI', 'AJG', 'APD', 'CHD', 'CVX', 'DLR', 'DVA', 'EQT', 'ETN', 'HBI', 'HES', 'INTU', 'IT', 'KMI', 'L', 'MAR', 'MET', 'MMM', 
    # 'NAVI', 'NOC', 'NSC', 'PLD', 'SLG', 'SPGI', 'SRCL', 'TJX', 'TMO', 'TSCO', 'TWX', 'UAA']
    # s_and_p = ['ADI', 'AJG', 'APD', 'CVX', 'DLR', 'DVA', 'ETN', 
    # 'HES', 'INTU', 'IT','L', 'MAR', 'MET', 'MMM', 'NOC', 'NSC', 'PLD', 'SPGI', 'TJX', 'TMO']

    s_and_p = ['TMO']
    
    for stock in s_and_p:
        env = MarketEnv(dir_path="./split_data/train/", target_codes=stock, sudden_death_rate=0.3, finalIndex = 997) #1259
        pg = DeepQ(env, gamma=0.80, model_file_name="./model/stock_model/model_"+stock+".h5")
        pg.train()

    # reward_stock = []
    # reward_stock_random = []

    # for stock in s_and_p:
    #     env = MarketEnv(dir_path="./split_data/test/", target_codes=stock, sudden_death_rate=0.3, finalIndex = 256)
    #     test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/stock_model/model_"+stock+".h5", test = True)
    #     test_obj_random = DeepQ(env, gamma=0.80, model_file_name="./model/stock_model/model_"+stock+".h5", test = True, random = True)


    #     reward_collect = []
    #     reward_collect_random = []

    #     for i in range(20):
    #         each_reward = test_obj.predict() 
    #         reward_collect.append(each_reward)
    #         each_reward = test_obj_random.predict()
    #         reward_collect_random.append(each_reward)

    #     reward_stock.append(np.mean(reward_collect))
    #     reward_stock_random.append(np.mean(reward_collect_random))

    # plt.plot(s_and_p,reward_stock)
    # plt.plot(s_and_p,reward_stock_random)
    # plt.title("Rewards Algorithm vs random")
    # plt.xlabel("stock")
    # plt.ylabel("return")
    # plt.legend(['DQN', 'Random'], loc='upper left')
    # plt.show()

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

















