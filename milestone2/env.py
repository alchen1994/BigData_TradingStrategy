from random import random
import numpy as np
import math

import gym
from gym import spaces


class MarketEnv(gym.Env):

    def __init__(self, dir_path, target_codes, sudden_death_rate=0.5, current_asset = {'cash':200, 'stock':0}, finalIndex = 1997):

        self.sudden_death_rate = sudden_death_rate
        self.current_asset = current_asset
        self.startAssetValue = 200
        self.baseAmount = 2
        self.current_trading_price = 0
        self.last_trading_price = 0
        self.code = target_codes
        self.data = []
        self.trading_feeRate = 0.0001
        self.finalIndex = finalIndex
        self.current_asset_value = 0
        self.last_asset_value = 0


        fn = dir_path + self.code + ".csv"

        lastDate = 0
        lastOpen = 0
        lastHigh = 0
        lastLow = 0
        lastClose = 0
        lastVolume = 0


        f = open(fn, "r")
        for line in f:
            if line.strip() != "":
                dt, openPrice, high, low, close, volume = line.strip().split(",")

                date = self.convertDate(dt)
                openPrice = float(openPrice)
                high = float(high) if high != "" else float(close)
                low = float(low) if low != "" else float(close)
                close = float(close)
                volume = float(volume)
                trading_price = (low+high+close)/3
                

                if lastClose > 0 and close > 0 and lastVolume > 0 and lastDate > 0:
                    date_ = (date - lastDate) / lastDate
                    open_ = (openPrice - lastOpen) / lastOpen
                    high_ = (high - lastHigh) / lastHigh
                    low_ = (low - lastLow) / lastLow
                    close_ = (close - lastClose) / lastClose
                    volume_ = (volume - lastVolume) / lastVolume
                    
                    self.data.append([date_, open_, high_, low_, close_, trading_price, volume_])
                    # self.data.append([date_, open_, high_, low_, close_, volume_])

                lastDate = date
                lastOpen = openPrice
                lastHigh = high
                lastLow = low
                lastClose = close
                lastVolume = volume

        f.close()


        self.actions = [
            "Buy",
            "Sell",
            "Hold",
        ]

        self.action_space = spaces.Discrete(len(self.actions))


        self._reset()


    def convertDate(self, date):
        date = date.split('-')
        date = date[1] + '.' + date[2]
        date = float(date)
        return date


    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.last_trading_price = float(self.data[self.currentIndex][5])
        self.current_trading_price = self.last_trading_price


        self.last_asset_value = round(float(self.current_asset['stock']) * self.last_trading_price + float(self.current_asset['cash']), 3)
        self.current_asset_value = self.last_asset_value 

        price_change = float(self.data[self.currentIndex+2][5]) - self.current_trading_price

        if self.actions[action] == "Buy":
            cur_stock = self.current_asset['stock']
            cur_cash = float(self.current_asset['cash'])

            self.current_asset['stock'] = round(cur_stock+self.baseAmount,3)
            self.current_asset['cash'] = round(cur_cash-float(self.current_trading_price)*self.baseAmount \
            - self.trading_feeRate * float(self.current_trading_price)*self.baseAmount,3)

            self.defineState()

            self.currentIndex += 1
            self.current_trading_price = float(self.data[self.currentIndex][5])

            self.current_asset_value = round((float(self.current_asset['stock']) * self.current_trading_price + float(self.current_asset['cash'])), 3)
            
            self.reward = self.current_asset_value - self.last_asset_value + price_change*self.baseAmount + 10 - self.current_asset['stock']


        elif self.actions[action] == "Sell":
            cur_cash = float(self.current_asset['cash'])
            cur_stock = self.current_asset['stock']
            
            self.current_asset['cash'] = round(self.baseAmount * float(self.current_trading_price) + cur_cash \
            - self.trading_feeRate * self.baseAmount* float(self.current_trading_price),3)
            self.current_asset['stock'] = round(cur_stock - self.baseAmount,3)

            self.defineState()
            self.currentIndex += 1
            self.current_trading_price = float(self.data[self.currentIndex][5])

            self.current_asset_value = round(float(self.current_asset['stock']) * self.current_trading_price+ float(self.current_asset['cash']), 3)

            self.reward = self.current_asset_value - self.last_asset_value - price_change*self.baseAmount + self.current_asset['stock'] - 7


        elif self.actions[action] == "Hold":
            self.defineState()
            self.currentIndex += 1

            self.reward = - 20
        else:
            pass

        if self.currentIndex >= self.finalIndex \
                or self.current_asset_value < float(self.sudden_death_rate * self.startAssetValue):

            self.done = True


        return self.state, self.reward, self.done, \
               {"date": self.currentIndex, "baseAmount": self.baseAmount,"trading_feeRate": self.trading_feeRate,\
                "current_trading_price": self.current_trading_price, \
                "current_asset":self.current_asset, "cur_reward": self.reward,
                "act":self.actions[action], "last_asset_value":self.last_asset_value, "current_asset_value":self.current_asset_value}


    def _reset(self):
        self.current_asset = {'cash':200, 'stock':0}

        self.currentIndex = 0

        self.done = False
        self.reward = 0

        self.current_trading_price = 0
        self.last_trading_price = 0

        self.current_asset_value = 200
        self.last_asset_value = 200


        self.defineState()


        return self.state


    def _render(self):
        return self.state


    def defineState(self):
        tmpState = []

        self.current_trading_price = self.data[self.currentIndex][5]

        cashFeature = float(self.current_asset['cash']) / 200
        stockFeature = float(self.current_asset['stock']) * float(self.current_trading_price) / 200
        
        tmpState = self.data[self.currentIndex].copy()

        tmpState.append(cashFeature)
        tmpState.append(stockFeature)

        
        tmpState = np.array(tmpState)
        tmpState = tmpState.reshape([1, tmpState.shape[0]])

        self.state = tmpState


    def printState(self):

        print (self.state)




