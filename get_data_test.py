#coding = utf-8
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import pdb
import h5py
pd.set_option('display.max_columns',None)

class data_entrance():
    def __init__(self):
        self.mainpath = r'F:\Data'
        self.tickpath =r'F:\Data\tickData'
        self.tradeday = pd.read_csv(r'C:\Users\z\Desktop\lstm-project\tradeday_series.csv')

    def get_tickData(self, date, stock):

        path = self.tickpath +'\\' + str(date) + '\\' + stock + '.mat'
        data = sio.loadmat(path)
        if len(data['tick']) > 0:
            tick = data['tick'][0][0]
        else:
            return pd.DataFrame()
        if isinstance(tick, str):
            return pd.DataFrame()
        df = pd.DataFrame({'Time': tick[2].reshape(-1), 'Price': tick[3].reshape(-1), 'Volume': tick[4].reshape(-1),
                           'Turnover': tick[23].reshape(-1), 'High': tick[10].reshape(-1), 'Low': tick[11].reshape(-1),
                           'MatchItems': tick[5].reshape(-1),
                           'AskAvPrice': tick[18].reshape(-1), 'BidAvPrice': tick[19].reshape(-1),
                           'TotalAskVolume': tick[20].reshape(-1), 'TotalBidVolume': tick[21].reshape(-1)})
        df['Date'] = tick[1][0, 0]
        df['Open'] = tick[12][0, 0]
        df['PreClose'] = tick[13][0, 0]
        # df['AskPrice1'] = tick[14][:, 0]
        # df['AskPrice2'] = tick[14][:, 1]
        # df['AskPrice3'] = tick[14][:, 2]
        # df['AskPrice4'] = tick[14][:, 3]
        # df['AskPrice5'] = tick[14][:, 4]
        # df['AskPrice6'] = tick[14][:, 5]
        # df['AskPrice7'] = tick[14][:, 6]
        # df['AskPrice8'] = tick[14][:, 7]
        # df['AskPrice9'] = tick[14][:, 8]
        # df['AskPrice10'] = tick[14][:, 9]

        for i in range(10):
            df['AskPrice'+str(i+1)] = tick[14][:,i]
        for i in range(10):
            df['AskVolume'+str(i+1)] = tick[15][:,i]
        for i in range(10):
            df['BidPrice'+str(i+1)] = tick[16][:,i]
        for i in range(10):
            df['BidVolume'+str(i+1)] = tick[17][:,i]

        df = df.loc[:,
             ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'MatchItems', 'High', 'Low', 'AskAvPrice', 'BidAvPrice',
              'Open', 'PreClose', 'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskPrice6', 'AskPrice7', 'AskPrice8', 'AskPrice9', 'AskPrice10',
              'AskVolume1','AskVolume2','AskVolume3','AskVolume4','AskVolume5','AskVolume6','AskVolume7','AskVolume8','AskVolume9','AskVolume10',
              'BidPrice1', 'BidPrice2','BidPrice3','BidPrice4','BidPrice5','BidPrice6','BidPrice7','BidPrice8','BidPrice9','BidPrice10',
              'BidVolume1','BidVolume2','BidVolume3','BidVolume4','BidVolume5','BidVolume6','BidVolume7','BidVolume8','BidVolume9','BidVolume10',
              'TotalAskVolume', 'TotalBidVolume']]
        df['Volume'] = df['Volume'].astype(np.float64)
        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def add_indextick(self,tickdata, indextick):
        indextick = indextick[indextick['Time'] < tickdata.loc[len(tickdata) - 1, 'Time']]
        pos = list(map(lambda x: np.where(np.array(tickdata['Time']) >= x)[0][0], indextick['Time']))

        tickdata.loc[pos, 'indexprice'] = indextick['Price'].values
        tickdata.loc[pos, 'indexvolume'] = indextick['Volume'].values
        tickdata.loc[pos, 'indexhigh'] = indextick['High'].values
        tickdata.loc[pos, 'indexlow'] = indextick['Low'].values
        pdb.set_trace()
        tickdata.loc[:, ['indexprice', 'indexvolume', 'indexhigh', 'indexlow']] = \
            tickdata.loc[:, ['indexprice', 'indexvolume', 'indexhigh', 'indexlow']].fillna(method='ffill')
        tickdata['indexPreClose'] = indextick.loc[0, 'PreClose']
        tickdata['indexOpen'] = indextick.loc[0, 'Open']
        tickdata['indexopenzdf'] = tickdata['indexOpen'] / tickdata['indexPreClose'] - 1
        return tickdata

df = data_entrance().get_tickData(20190426,'000025.SZ')
print(df.head(5))
df.to_csv('test.csv')
