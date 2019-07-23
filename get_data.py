# coding = utf-8
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import pdb
import h5py

class data_entrance():
    def __init__(self):
        self.mainpath = 'I:\HFdata'
        self.callautionpath = self.mainpath + r'\call_pre_market'
        self.tickpath = self.mainpath + r'\tick_eachstock'
        self.tranpath = self.mainpath + r'\tran_eachstock'
        self.orderQpath = self.mainpath + r'\orderQ_eachstock'
        self.indexpath = self.mainpath + r'\index'
        self.futurepath = self.mainpath + r'\cf_tick'
        self.minpath = self.mainpath + r'\mindata'
        self.tradeday = pd.read_csv('..\\tradeday_series.csv')

    def getcodelist(self,date):
        listname = os.listdir(self.tranpath + '\%s'%date)
        codelist = []
        for name in listname:
            name = name[:9]
            codelist.append(name)
        return codelist

    def getindexcodelist(self,date):
        listname = os.listdir(self.indexpath + '\%s'%date)
        codelist = []
        for name in listname:
            name = name[:9]
            codelist.append(name)
        return codelist

    def get_callaution(self,date,stock):
        '''
        读取集合竞价tick数据，一般只需要'AskPrice','AskVolume','BidPrice','BidVolume'
        :param path: 单个股票tick数据路径，格式为'../.../stock.mat'
        :return: 返回tick数据  ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'High', 'Low',
                                'Open','Close','PreClose','AskPrice','AskVolume','BidPrice','BidVolume']
                    对应数据为 [日期，tick时间，价格，成交量，成交额，tick中的最高价，tick中的最低价，
                                今开，今日，昨收，卖一至卖十的价格，对应的量，买一至买十的价格，对应的量]
                    其中 'AskPrice','AskVolume','BidPrice','BidVolume' 均为array的格式
        '''
        path = self.callautionpath + '\\' + str(date) + '\\' + stock + '.mat'
        data = sio.loadmat(path)
        if len(data['tick']) > 0:
            tick = data['tick'][0][0]
        else:
            return pd.DataFrame()
        if isinstance(tick, str):
            # print('no data')
            return pd.DataFrame()
        df = pd.DataFrame({'Time': tick[2].reshape(-1), 'Price': tick[3].reshape(-1), 'Volume': tick[4].reshape(-1),
                           'Turnover': tick[23].reshape(-1), 'High': tick[10].reshape(-1), 'Low': tick[11].reshape(-1),
                           'AccVolume': tick[8].reshape(-1), 'AccTurnover': tick[9].reshape(-1),
                           'MatchItems': tick[5].reshape(-1),
                           'AskAvPrice': tick[18].reshape(-1), 'BidAvPrice': tick[19].reshape(-1),
                           'TotalAskVolume': tick[20].reshape(-1), 'TotalBidVolume': tick[21].reshape(-1)})
        # df['Code'] = tick[0][0]
        df['Date'] = tick[1][0, 0]
        df['Open'] = tick[12][0, 0]
        df['PreClose'] = tick[13][0, 0]
        # df['Close'] = tick[21][0, 0]
        df['AskPrice'] = list(tick[14])
        df['AskVolume'] = list(tick[15])
        df['BidPrice'] = list(tick[16])
        df['BidVolume'] = list(tick[17])
        df = df.loc[:,
             ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'MatchItems', 'High', 'Low', 'AskAvPrice', 'BidAvPrice',
              'Open', 'PreClose', 'AskPrice', 'AskVolume', 'BidPrice', 'BidVolume', 'TotalAskVolume', 'TotalBidVolume',
              'AccVolume', 'AccTurnover']]
        df['Volume'] = df['Volume'].astype(np.float64)
        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def get_tickdata(self,date,stock):
        '''
        读取tick数据
        :param path: 单个股票tick数据路径，格式为'../.../stock.mat'
        :return: 返回tick数据  ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'High', 'Low',
                                'Open','Close','PreClose','AskPrice','AskVolume','BidPrice','BidVolume']
                    对应数据为 [日期，tick时间，价格，成交量，成交额，tick中的最高价，tick中的最低价，
                                今开，今日，昨收，卖一至卖十的价格，对应的量，买一至买十的价格，对应的量]
                    其中 'AskPrice','AskVolume','BidPrice','BidVolume' 均为array的格式
        '''
        path = self.tickpath + '\\' + str(date) + '\\' + stock + '.mat'
        data = sio.loadmat(path)
        if len(data['tick'])>0:
            tick = data['tick'][0][0]
        else:
            return pd.DataFrame()
        if isinstance(tick, str):
            # print('no data')
            return pd.DataFrame()
        df = pd.DataFrame({'Time': tick[2].reshape(-1), 'Price': tick[3].reshape(-1), 'Volume': tick[4].reshape(-1),
                           'Turnover': tick[23].reshape(-1), 'High': tick[10].reshape(-1), 'Low': tick[11].reshape(-1),
                           'AccVolume':tick[8].reshape(-1),'AccTurnover':tick[9].reshape(-1),'MatchItems':tick[5].reshape(-1),
                           'AskAvPrice':tick[18].reshape(-1),'BidAvPrice':tick[19].reshape(-1),
                           'TotalAskVolume':tick[20].reshape(-1),'TotalBidVolume':tick[21].reshape(-1)})
        # df['Code'] = tick[0][0]
        df['Date'] = tick[1][0, 0]
        df['Open'] = tick[12][0, 0]
        df['PreClose'] = tick[13][0, 0]
        # df['Close'] = tick[21][0, 0]
        df['AskPrice'] = list(tick[14])
        df['AskVolume'] = list(tick[15])
        df['BidPrice'] = list(tick[16])
        df['BidVolume'] = list(tick[17])
        df = df.loc[:,
             ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'MatchItems', 'High', 'Low', 'AskAvPrice', 'BidAvPrice',
              'Open', 'PreClose', 'AskPrice', 'AskVolume', 'BidPrice', 'BidVolume', 'TotalAskVolume', 'TotalBidVolume',
              'AccVolume', 'AccTurnover']]
        df['Volume'] = df['Volume'].astype(np.float64)
        # 存在重复时间戳且信息不冗余情形，暂时不处理
        # # 重复时间剔除，由于信息的冗余造成的,多余的是volume=0的一个
        # dup = df.groupby('Time')['Time'].count()
        # dup = dup[dup>1]
        # delete_index = []
        # for t in dup.index:
        #     try:
        #         delete_index.append(df[(df['Time']==t) & (df['Volume'] == 0)].index[0])
        #     except:
        #         print('存在重复时间戳，但无信息冗余')
        # df = df.drop(delete_index)

        # 将Price = 0 的数据按前值补，再按昨收补
        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def get_trandata(self, date, stock):
        '''
        读取逐笔数据
        :param path: 逐笔数据的路径，格式为'../.../stock.mat'
        :return: 返回逐笔数据['Date','Time','Price','TradeVolume','AskOrder','BidOrder','BSFlag']
                    对应为[日期，成交时间，成交价格，成交股数，卖单号，买单号，买卖方向（B/S）]
        '''
        path = self.tranpath + '\\' + str(date) + '\\' + stock + '.mat'
        data = sio.loadmat(path)
        tran = data['tran'][0][0]
        if isinstance(tran, str):
            # print('no data')
            return pd.DataFrame()
        df = pd.DataFrame(
            {'Time': tran[2].reshape(-1), 'Price': tran[7].reshape(-1), 'TradeVolume': tran[8].reshape(-1),
             'AskOrder': tran[9].reshape(-1), 'BidOrder': tran[10].reshape(-1),
             'BSFlag': tran[6].reshape(-1)})
        # df['Code'] = tick[0][0]
        df['Date'] = tran[1][0][0]
        df['BSFlag'] = df['BSFlag'].apply(lambda x: 1 if x == 'B' else -1)  # 1表示Buy; -1表示Sell
        df['TradeVolume'] = df['TradeVolume'].astype(np.float64)
        df = df.loc[:, ['Date', 'Time', 'Price', 'TradeVolume', 'AskOrder', 'BidOrder', 'BSFlag']]
        return df

    def merge_ticktran(self,tickdata, trandata):
        '''
        将tick数据和tran数据匹配
        :param tickdata: tick数据
        :param trandata: 逐笔数据
        :return:trandata, tickdata, matchlist, errorflag
                trandata:  新增列ticknum，表示所在tick的序号
                tickdata:  不变
                matchlist: 匹配的列表，startidx为该tick对应trandata开始的位置，endidx对应trandata结束的位置
                errorflag: 错误标记，若为1，则匹配失败
        '''
        errorflag = 0  # 出错提示
        # 先将tran数据中Price =0 的部分剔除，该部分为撤单部分，且此时TradeVolume可能不为0,使其无法与tick对应
        trandata = trandata[trandata['Price'] != 0]
        trandata = trandata.reset_index(drop=True)
        # 利用累计成交量对应的方法将其合并
        trandata['accv'] = trandata['TradeVolume'].cumsum()
        tickdata['accv'] = tickdata['Volume'].cumsum()
        tranindex = trandata.loc[:, ['accv']]
        tranindex['endidx'] = tranindex.index
        matchlist = tickdata.loc[:, ['accv']].merge(tranindex, how='left', on='accv')
        if tickdata['accv'].values[-1] != trandata['accv'].values[-1]:
            # 若存在index有空的情形，说明此时有不匹配的情况,有可能为第一个tick没有成交
            errorflag = 1
        else:
            # 生成一个匹配的列表，startidx为该tick对应trandata开始的位置，endidx对应trandata结束的位置
            matchlist.loc[tickdata['Volume'] == 0, 'endidx'] = np.nan
            startidx = matchlist.loc[tickdata['Volume'] != 0, 'endidx'].values + 1
            matchlist.loc[tickdata['Volume'] != 0, 'startidx'] = np.hstack([0, startidx[:-1]])
            matchlist = matchlist.loc[:, ['startidx', 'endidx']]
            ## 将trandata用tick的序号标记
            trandata.loc[trandata.index.isin(matchlist['startidx'].values), 'ticknum'] = tickdata[tickdata['Volume'] != 0].index
            trandata['ticknum'] = trandata['ticknum'].fillna(method='ffill')
        return (tickdata, trandata, matchlist, errorflag)

    def get_tickwithtran(self,date, stock):
        '''
        获取tick数据及其对应的逐笔数据
        :param date:
        :param stock:
        :param trimdata:
        :return:
        '''
        tickdata = self.get_tickdata(date,stock)
        trandata = self.get_trandata(date,stock)

        if len(tickdata) != 0 and len(trandata) != 0:  # 若不存在数据会返回空的dataframe
            tickdata,trandata,matchlist, errorflag = self.merge_ticktran(tickdata, trandata)
            if errorflag == 1:
                print('Matching Failed, please check the data!')
            return tickdata,trandata,matchlist,errorflag
        else:
            return tickdata,trandata,None,0

    def get_tickwithorderQ(self, date, stock):
        '''
        获取tick数据及其对应的逐笔数据
        :param date:
        :param stock:
        :param trimdata:
        :return:
        '''
        tickdata = self.get_tickdata(date, stock)
        orderQdata = self.getorderQ(date, stock)

        if len(tickdata) != 0 :  # 若不存在数据会返回空的dataframe
            tickdata = tickdata.merge(orderQdata, how='left', on=['Date', 'Time'])
            return tickdata
        else:
            return pd.DataFrame()

    def get_completetick(self,date,stock):
        df1 = self.get_callaution(date,stock)
        df2 = self.get_tickdata(date,stock)
        df1 = df1[df1['Time']<92500000]
        df = pd.concat([df1,df2])
        # pdb.set_trace()
        # df = df.drop_duplicates(subset=['Date','Time'])
        return df

    def getindex(self,date,indexcode):
        '''
        读取tick数据
        :param path: 单个股票tick数据路径，格式为'../.../stock.mat'
        :return: 返回tick数据  ['Date', 'Time', 'Price', 'Volume', 'Turnover', 'High', 'Low',
                                'Open','Close','PreClose','AskPrice','AskVolume','BidPrice','BidVolume']
                    对应数据为 [日期，tick时间，价格，成交量，成交额，tick中的最高价，tick中的最低价，
                                今开，今日，昨收，卖一至卖十的价格，对应的量，买一至买十的价格，对应的量]
                    其中 'AskPrice','AskVolume','BidPrice','BidVolume' 均为array的格式
        '''
        path = self.indexpath + '\\' + str(date) + '\\' + indexcode + '.mat'
        data = sio.loadmat(path)
        tick = data['tickdata_temp'][0][0]
        if isinstance(tick, str):
            # print('no data')
            return pd.DataFrame()
        try:
            df = pd.DataFrame({'Time': tick[3].reshape(-1), 'Price': tick[4].reshape(-1), 'Volume': tick[5].reshape(-1),
                               'Turnover': tick[6].reshape(-1), 'High': tick[13].reshape(-1), 'Low': tick[14].reshape(-1),
                               'Open': tick[15].reshape(-1), 'PreClose': tick[16].reshape(-1)})
        except:
            pdb.set_trace()
        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def getindex2(self, date, indexcode):
        '''
        v7.3的应用hdf来读取数据
        :param date:
        :param indexcode:
        :return:
        '''
        path = self.indexpath + '\\' + str(date) + '\\' + indexcode + '.mat'
        HDF5_file = h5py.File(path, 'r')
        data = HDF5_file['tickdata_temp']
        if 'tickdata_temp' in data:
            data = data['tickdata_temp']
        df = pd.DataFrame({'Time': data['Time'].value[0],'Price': data['Price'].value[0],'Volume':data['Volume'].value[0],
                            'Turnover':data['Turover'].value[0],'High':data['High'].value[0], 'Low':data['Low'].value[0],
                           'Open':data['Open'].value[0],'PreClose':data['PreClose'].value[0]})
        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def getfuture(self, date, code, Num = 'first'):
        dfloc = pd.read_excel(self.futurepath + r'\future_info_%s.xlsx'%code,header=None)
        dfloc.columns = ['date','first','second','third', 'fourth']
        fcode =dfloc.loc[dfloc['date'] == date, Num].values[0]
        path = self.futurepath + '\\' + str(date) + '\\' + fcode + '.mat'
        data = sio.loadmat(path)
        tick = data['tick'][0][0]
        if isinstance(tick, str):
            # print('no data')
            return pd.DataFrame()
        try:
            df = pd.DataFrame({'Time': tick[3].reshape(-1), 'Price': tick[4].reshape(-1), 'Volume': tick[5].reshape(-1),
                               'Turnover': tick[6].reshape(-1), 'High': tick[13].reshape(-1),
                               'Low': tick[14].reshape(-1),
                               'Open': tick[15].reshape(-1), 'PreClose': tick[16].reshape(-1)})
        except:
            pdb.set_trace()
        df['AskPrice'] = list(tick[22])
        df['AskVolume'] = list(tick[23])
        df['BidPrice'] = list(tick[24])
        df['BidVolume'] = list(tick[25])

        df['Price'] = df['Price'].replace(0, np.nan)
        df['Price'] = df['Price'].fillna(method='ffill')
        df['Price'] = df['Price'].fillna(df['PreClose'])
        df = df.reset_index(drop=True)
        return df

    def getorderQ(self,date,stock):
        '''
        获取委托队列数据，队列为空的时候说明数据没有变化
        :param date:
        :param stock:
        :return:
        '''
        path = self.orderQpath + '\\' + str(date) + '\\' + stock + '.mat'
        data = sio.loadmat(path)
        orderQ = data['orderQ'][0][0]
        if isinstance(orderQ, str):
            # print('no data')
            return pd.DataFrame()
        df = pd.DataFrame(
            {'Time': orderQ[2].reshape(-1), 'Price': orderQ[4].reshape(-1), 'Side': orderQ[3].reshape(-1),
             'abitems': orderQ[6].reshape(-1), 'Orderitems': orderQ[5].reshape(-1),})
        # df['Code'] = tick[0][0]
        df['abvolume'] = list(orderQ[7])
        dfA = df[df['Side'] == 'A']
        dfB = df[df['Side'] == 'B']
        dfA.rename(columns={'Price': 'A1Price', 'abitems': 'A1items', 'Orderitems': 'A1allitem', 'abvolume': 'A1volume'},
                   inplace=True)
        dfB.rename(columns={'Price': 'B1Price', 'abitems': 'B1items', 'Orderitems': 'B1allitem', 'abvolume': 'B1volume'},
                   inplace=True)
        df = dfA.merge(dfB,how='left',on='Time')
        df['Date'] = orderQ[1][0][0]
        df = df.loc[:, ['Date', 'Time', 'A1Price', 'B1Price', 'A1items', 'B1items', 'A1allitem', 'B1allitem', 'A1volume', 'B1volume']]
        return df

    def add_indextick(self,tickdata, indextick):
        indextick = indextick[indextick['Time'] < tickdata.loc[len(tickdata) - 1, 'Time']]
        pos = list(map(lambda x: np.where(np.array(tickdata['Time']) >= x)[0][0], indextick['Time']))

        tickdata.loc[pos, 'indexprice'] = indextick['Price'].values
        tickdata.loc[pos, 'indexvolume'] = indextick['Volume'].values  # 采用累计成交量
        tickdata.loc[pos, 'indexhigh'] = indextick['High'].values
        tickdata.loc[pos, 'indexlow'] = indextick['Low'].values
        pdb.set_trace()
        tickdata.loc[:, ['indexprice', 'indexvolume', 'indexhigh', 'indexlow']] = \
            tickdata.loc[:, ['indexprice', 'indexvolume', 'indexhigh', 'indexlow']].fillna(method='ffill')
        tickdata['indexPreClose'] = indextick.loc[0, 'PreClose']
        tickdata['indexOpen'] = indextick.loc[0, 'Open']
        tickdata['indexopenzdf'] = tickdata['indexOpen'] / tickdata['indexPreClose'] - 1
        return tickdata

    def get_tickdata_serial(self, startdate, enddate, stock):
        '''
        获取多天tick数据
        :param startdate:开始日期
        :param enddata: 结束日期
        :param stock: 股票代码
        :return:
        '''
        _getdays = self.tradeday
        _getdays = _getdays[(_getdays['date'] >= startdate) & (_getdays['date'] <= enddate)]
        _getdays = _getdays.reset_index(drop=True)
        tickdata_serial = pd.DataFrame()
        for i in range(len(_getdays)):
            _tick = self.get_tickdata(_getdays.loc[i, 'date'], stock)
            tickdata_serial = pd.concat([tickdata_serial, _tick])
        tickdata_serial = tickdata_serial.reset_index(drop=True)
        return tickdata_serial

    def transform_to_min(self, tickdata, setmin=1):
        '''
        将tick数据转换成分钟数据
        :param tickdata:
        :param setmin: 目标数据是n分钟的数据
        :return:
        '''
        starttime = 93000000
        min_series = pd.read_excel('..\min_series.xlsx')
        min_series = min_series / 1000
        tickdata = tickdata.loc[tickdata['Time'] >= starttime, :]
        tickdata.loc[:, 'minute'] = tickdata['Time'].map(lambda x: int(x / 100000) * 100)
        tickdata.loc[(tickdata['minute'] >= 113000) & (tickdata['minute'] <= 125900), 'minute'] = 112900
        tickdata.loc[tickdata['minute'] > 145900, 'minute'] = 145900
        tickdata = min_series.merge(tickdata, how='left', on='minute')
        # tickdata2 = tickdata[tickdata['Volume'] != 0]
        mintemp = tickdata.groupby('minute')
        mint = mintemp['minute'].first()
        open = mintemp['Price'].first()
        close = mintemp['Price'].last()
        high = mintemp['Price'].max()
        low = mintemp['Price'].min()
        vol = mintemp['Volume'].sum()
        amt = mintemp['Turnover'].sum()
        mindata = pd.DataFrame({'minute': mint, 'open': open, 'high': high,'low': low,  'close': close,
                                'volume': vol,'turnover': amt})
        # mindata['preclose'] = mindata['close'].shift(1)
        # pdb.set_trace()
        # tickdata[tickdata['minute']==93200]
        # mindata.loc[:,['open','high','low','close','preclose']] = mindata.loc[:,['open','high','low','close','preclose']].replace(0,np.nan)
        # mindata['open'] = mindata['open'].fillna(mindata['preclose'])
        # mindata['close'] = mindata['close'].fillna(mindata['preclose'])
        # mindata['high'] = mindata['high'].fillna(mindata['preclose'])
        # mindata['low'] = mindata['low'].fillna(mindata['preclose'])
        if setmin > 1:
            mindata.loc[mindata.index % setmin == 0, 'mark'] = 1
            mindata['mark'] = mindata['mark'].cumsum()
            mindata['mark'].fillna(method='ffill')
            # 开收高低
            mindata = mindata.groupby('mark').agg({'minute': 'first', 'open': 'first', 'close': 'last',
                                                   'high': 'max', 'low': 'min', 'volume': 'sum', 'turnover': 'sum'})
        mindata = mindata.reset_index(drop=True)
        return tickdata, mindata

    def get_mindata_serial(self, startdate, enddate, stock, setmin=1, returntick=False):
        '''
        获取用tick数据合成的分钟数据
        :param startdate: 开始日期
        :param endate: 结束日期
        :param stock: 股票代码
        :param setmin 划分分钟单位
        :param returntick: 是否返回对应的tick数据
        :return:
        '''
        _getdays = self.tradeday
        _getdays = _getdays[(_getdays['date'] >= startdate) & (_getdays['date'] <= enddate)]
        _getdays = _getdays.reset_index(drop=True)
        tickdata_serial = pd.DataFrame()
        mindata_serial = pd.DataFrame()
        for i in range(len(_getdays)):
            _tick = self.get_tickdata(_getdays.loc[i, 'date'], stock)
            if len(_tick)>0:
                if returntick == True:
                    _tick, mindata = self.transform_to_min(_tick)
                    tickdata_serial = pd.concat([tickdata_serial, _tick])
                else:
                    _, mindata = self.transform_to_min(_tick,setmin)
                mindata['date'] = _getdays.loc[i, 'date']
            else:
                mindata = pd.DataFrame()
            mindata_serial = pd.concat([mindata_serial, mindata])
        mindata_serial = mindata_serial.reset_index(drop=True)
        if returntick == True:
            return mindata_serial,tickdata_serial
        else:
            return mindata_serial

    def get_mindata_serial_local(self,startdate,enddate,stock,setmin=1):
        _getdays = self.tradeday
        _getdays = _getdays[(_getdays['date'] >= startdate) & (_getdays['date'] <= enddate)]
        _getdays = _getdays.reset_index(drop=True)
        mindata_all = pd.DataFrame()
        for i in range(len(_getdays)):
            try:
                mindata = pd.read_csv(self.minpath + '\%s\%s.csv' %(_getdays.loc[i,'date'],stock))
                if setmin > 1:
                    mindata.loc[mindata.index % setmin == 0, 'mark'] = 1
                    mindata['mark'] = mindata['mark'].cumsum()
                    mindata['mark'].fillna(method='ffill')
                    # 开收高低
                    mindata = mindata.groupby('mark').agg({'minute': 'first', 'open': 'first', 'close': 'last',
                                                           'high': 'max', 'low': 'min', 'volume': 'sum',
                                                           'turnover': 'sum'})
                mindata['date'] = _getdays.loc[i, 'date']
                mindata = mindata.reset_index(drop=True)
            except:
                mindata = pd.DataFrame()
            mindata_all = pd.concat([mindata_all, mindata])
        mindata_all = mindata_all.reset_index(drop=True)
        return mindata_all

    def rolling_n_min(self,tickdata, window):
        '''
        往前或往后寻找n分钟以前的index
        :param tickdata:
        :param window: 移动分钟窗口，其中大于0时往前寻找，小于0时往后寻找
        :return:
        '''
        # 交易日内秒的数据，并与原本tick数据的秒数据进行合并
        # 由于数据可能由于发送延迟等原因会出现一些边角，可能无法匹配日内的秒数据，因此需要先合并处理
        second_series = pd.read_excel('..\second_series.xlsx')
        second_series = second_series.merge(tickdata.loc[:,['Time']],how='outer',on='Time')
        second_series = second_series[second_series['Time']>=93000000]
        second_series = second_series.sort_values('Time')
        second_series = second_series.reset_index(drop=True)
        # 找到日内秒数据中对应tick数据的index
        ticktime = tickdata.loc[:,['Time']]
        ticktime['pos'] = ticktime.index
        second_series = second_series.merge(ticktime,how='left',on='Time')
        second_series['pos'] = second_series['pos'].fillna(method='bfill')
        # 往前/后滚动数第60*n秒
        span = window * 60  # 1分钟等于60秒
        second_series['loc'] = second_series['pos'].shift(span)

        # 匹配对应的位置
        tickdata2 = tickdata.merge(second_series.loc[:,['Time','loc']],how='left',on='Time')
        return tickdata2['loc'].values

    def get_future_min_zf(self,tickdata, fmin):
        '''
        获取fmin分钟前/后的涨跌幅 若fmin为正则为往后涨幅，为负则为前面n分钟到该点的涨幅
        :param tickdata:
        :param fmin:
        :return:
        '''
        tickdata['pos'] = self.rolling_n_min(tickdata,fmin)
        # tickdata['return'] = list(map(lambda x,y: tickdata.loc[x,'Price']/y-1 if ~np.isnan(x) else np.nan,tickdata['pos'],tickdata['Price']))
        if fmin>0:
            tickdata.loc[~tickdata['pos'].isnull(),'return'] = \
                tickdata.loc[tickdata.loc[~tickdata['pos'].isnull(),'pos'].values,'Price'].values\
                /tickdata.loc[~tickdata['pos'].isnull(),'Price'].values-1
        else:
            tickdata.loc[~tickdata['pos'].isnull(), 'return'] = \
                tickdata.loc[~tickdata['pos'].isnull(), 'Price'].values \
                / tickdata.loc[tickdata.loc[~tickdata['pos'].isnull(), 'pos'].values, 'Price'].values - 1
        return tickdata['return'].values

    def get_min_vol(self, tickdata, fmin):
        '''
        获取fmin分钟前/后的收益率波动率 若fmin为正则为往后，为负则为前面n分钟到该点
        :param tickdata:
        :param fmin:
        :return:
        '''
        tickdata['pos'] = self.rolling_n_min(tickdata,fmin)
        # tickdata['return'] = list(map(lambda x,y: tickdata.loc[x,'Price']/y-1 if ~np.isnan(x) else np.nan,tickdata['pos'],tickdata['Price']))
        if fmin>0:
            tickdata.loc[~tickdata['pos'].isnull(),'return'] = \
                tickdata.loc[tickdata.loc[~tickdata['pos'].isnull(),'pos'].values,'Price'].values\
                /tickdata.loc[~tickdata['pos'].isnull(),'Price'].values-1
            tickdata['vol'] = tickdata['return'].ewm(span=100).std()
        else:
            tickdata.loc[~tickdata['pos'].isnull(), 'return'] = \
                tickdata.loc[~tickdata['pos'].isnull(), 'Price'].values \
                / tickdata.loc[tickdata.loc[~tickdata['pos'].isnull(), 'pos'].values, 'Price'].values - 1
            tickdata['vol'] = tickdata['return'].ewm(span=100).std()
        return tickdata['vol'].values

    def trimdata(self,df, trimdata=True, cutoff=False):
        if trimdata == True:
            df = df[df['Price'] != 0]
        if cutoff == True:
            df = df[(df['Time'] >= 93000000) & (df['Time'] <= 150000000)]