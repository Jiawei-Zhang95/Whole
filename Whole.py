from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
import pandas as pd
import numpy as np
import pdb
import warnings
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,TimeDistributed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
pd.set_option('display.max_columns', 10)

path =r'C:\Users\user\Desktop\my-LSTM-Project\Stock-LSTM-master\Stock-LSTM-master\000025.SZ'
operate_path =r'C:\Users\user\Desktop\my-LSTM-Project\Stock-LSTM-master\Stock-LSTM-master\testresult'

#计算每一个交易日的上涨，横盘，下跌样本数
#以及每一个交易日的上涨，横盘，下跌准确率
def precisionCalculate(pred_y, test_y,test_start_date):
    count = pred_y + test_y #pred_y 与 test_y的预测值都是0,1,2 （0：上涨，1：横盘，2：下跌）
    firstZero = len(count[count==0]) #两者预测都为上涨的个数
    intersectZero = np.sum(list(map(lambda x,y: (x==y==1) , pred_y, test_y)))#两者预测都为横盘的个数
    countFour = len(count[count == 4])#两者预测都为下跌的个数

    precision1 = firstZero / len(pred_y[pred_y==0] ) #precision1： 上涨准确率
    precision2 = intersectZero / len(pred_y[pred_y==1]) #precision2： 横盘准确率
    precision3 = countFour / len(pred_y[pred_y==2]) #precision3： 下跌准确率

    #预测样本
    #ratio1 = len(pred_y[pred_y==0]) / len(count)
    #ratio2 = len(pred_y[pred_y==1]) / len(count)
    #ratio3 = len(pred_y[pred_y==2])/len(count)

    #下跌样本
    test1 = len(test_y[test_y==0]) / len(count)
    test2 = len(test_y[test_y==1]) / len(count)
    test3 = len(test_y[test_y==2])/len(count)
    #打印出来
    print('Precision 1: %0.3f' %precision1)
    print('Precision 2: %0.3f' %precision2)
    print('Precision 3: %0.3f' %precision3)

    temp = [[test_start_date, test1, test2, test3, precision1, precision2, precision3]]
    df_temp = pd.DataFrame(temp,columns=['Date', '上涨样本', '横盘样本', '下跌样本', '上涨准确率', '横盘准确率', '下跌准确率'])
    return df_temp

#Early Stopping函数，暂时没有用到
class EarlyStoppingByLossVal(Callback):
    def __init__(self,monitor ='val_loss',value = 0.00001,verbose = 0):
        super(Callback, self).__init__()
        self.monitor =monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requies %s available' %self.monitor, RuntimeError)

        if current < self.value:
            if self.verbose >0:
                print('Epoch %d: early stopping THR' % epoch)
                self.model.stop_training =True


tradeday_series = pd.read_csv(r'C:\Users\user\Desktop\my-LSTM-Project\Stock-LSTM-master\Stock-LSTM-master\tradeday_series.csv',index_col=None)
start = 20190101 #训练集开始日期
end = 20190514 #测试集结束日期（测试集开始为 end - start + （train_day - valid_day））
train_day = 30
test_day = 1
valid_day = 6

#训练，验证，测试总数据
df_all = pd.DataFrame()

#样本比例和准确率.xlsx
df_result = pd.DataFrame(columns=['Date', '上涨样本', '横盘样本', '下跌样本', '上涨准确率', '横盘准确率', '下跌准确率'])

tradeday_series= tradeday_series[(tradeday_series['date']>= start) & (tradeday_series['date'] <= end)]
tradeday_series =tradeday_series.reset_index(drop=True)
for date in tradeday_series['date'].values:

    try:
        df = pd.read_csv(path + '/%s.csv'%date)
        df= df[100:len(df)-100] #去掉每一天的头尾100个交易
        df = df[df['Time'] <= 145000000] #只取14:50之前的数据
        df.drop(['Price','MidPrice','PreClose'],axis = 1, inplace = True)
    except:
        continue
    df_all = pd.concat([df_all,df])


#总数据预处理
df_all = df_all.reset_index(drop=True)
df_all = df_all.set_index(df_all['Date'])
df_all = df_all.fillna(0)
df_all[:] = np.nan_to_num(df_all)


for i in range(len(tradeday_series) - train_day - valid_day - test_day + 1):
    train_start_date = tradeday_series.loc[i,'date'] #训练开始
    train_end_date = tradeday_series.loc[i + train_day - 1, 'date'] #训练结束
    valid_start_date = tradeday_series.loc[i + train_day,'date'] #验证开始
    valid_end_date = tradeday_series.loc[i + train_day + valid_day - 1, 'date'] #验证结束
    test_start_date = tradeday_series.loc[i + train_day + valid_day, 'date'] # 测试开始
    test_end_date = tradeday_series.loc[i + train_day + valid_day + test_day - 1, 'date'] #测试结束

    #划分训练集，验证集和测试集
    train = df_all.ix[train_start_date:train_end_date]
    test = df_all.ix[test_start_date:test_end_date]
    valid = df_all.ix[valid_start_date:valid_end_date]

    #将日期和时间保存起来
    ttime = test.loc[:,'Time'].values
    ddate = test.loc[:,'Date'].values

    #将train_y往上移动一格。这么做的意义是同一行所有的feature对应的是 y+1 的值
    #即预测下一个tick所对应的特征
    train['bin'] = train['bin'].shift(-1)
    train= train.iloc[:-1, :]
    train = train.values
    test = test.values
    #划分训练，验证和测试集
    train_x = train[:,3:]
    train_y = train[:,2]
    test_x = test[:,3:]
    test_y = test[:,2]
    #验证集的预处理
    valid = valid.values
    valid_x = valid[:,3:]
    valid_y = valid[:,2]

    #数据正则化，将所有x变为0和1的区间
    scaler = MinMaxScaler(feature_range=(0,1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    valid_x = scaler.fit_transform(valid_x)


    #将训练，验证和测试转变为一个3D的数据集
    train_x = train_x.reshape((train_x.shape[0],1,train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0],1,test_x.shape[1]))
    valid_x = valid_x.reshape((valid_x.shape[0],1,valid_x.shape[1]))
    #改变所有的y（输出值）为[1,0,0][0,1,0][0,0,0]
    onehot_encoder = OneHotEncoder()
    max_ = train_y.max()
    max2 = test_y.max()
    max3 = valid_y.max()
    #将所有的y变为正数 （One hot encoder不支持负数）
    train_y = (train_y - max_) * (-1)
    test_y = (test_y - max2) * (-1)
    valid_y = (valid_y - max3) * (-1)
    encode_categorical = train_y.reshape(len(train_y), 1)
    encode_categorical2 = test_y.reshape(len(test_y), 1)
    encode_categorical3 = valid_y.reshape(len(valid_y), 1)
    #现在的y变成了[1,0,0][0,1,0][0,0,0]的形式
    train_y = onehot_encoder.fit_transform(encode_categorical).toarray()
    test_y = onehot_encoder.fit_transform(encode_categorical2).toarray()
    valid_y = onehot_encoder.fit_transform(encode_categorical3).toarray()

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape,valid_x.shape,valid_y.shape)
    #设置Early stopping中callbacks的参数

    model_weights_path = r'C:\Users\user\Desktop\my-LSTM-Project\Stock-LSTM-master\Stock-LSTM-master\model_best_weights\weights.best.hdf5'
    checkpoint = ModelCheckpoint(model_weights_path,monitor='val_loss',verbose=1,save_best_only=True)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2),checkpoint]
    batch_size = 128
    #初始化LSTM模型
    model = Sequential()
    model.add(LSTM(64,input_shape=(1,90),activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08,decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=10,validation_data=(valid_x,valid_y),callbacks=callbacks,shuffle=False,verbose=0,batch_size=batch_size)
    #模型评估
    scores,results = model.evaluate(test_x,test_y,verbose=0)
    #预测值pred_y
    pred_y = model.predict(test_x)
    #将预测值变为0,1,2
    pred_y = np.argmax(pred_y,axis=1)
    #将真实值变为0,1,2
    test_y_alter = np.argmax(test_y,axis=1)
    # 预测值为0,1,2个对应的概率
    pred_y_prob = model.predict_proba(test_x)
    # 打印出总体准确率
    print('Accuracy: %0.3f' % results)
    pdb.set_trace()
    #从现在开始创建第二个LSTM
    #train_x , test_x保持不变
    #train_y2 = model.predict(train_x) * test_y_temp
    #test_y2 = model.predict(test_x) & test_y_temp
    #y的结果为0或者4，为1。 y的结果为1或者2，为0
    #输出的label为0，1

    # 这是train_y2 和 test_y2都要乘的东西

    train_y_temp = np.argmax(train_y,1)
    test_y_temp = np.argmax(test_y,1)
    train_y2 = model.predict(train_x) #新的train_y2
    train_y2 = np.argmax(train_y2,1)
    test_y2 = pred_y #新的test_y2

    def shift(arr,num, fill_value = np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

    #train_y2 = shift(train_y2,-1,0)
    #乘以test_y_temp
    train_y2 = train_y2 * train_y_temp
    test_y2 = test_y2 * test_y_temp

    #哦对我忘记了验证集的x，y， 现在开始初始化验证集
    valid_y2 = model.predict(valid_x)
    valid_y2 = np.argmax(valid_y2,1)
    train_y2[(train_y2 == 0) | (train_y2 == 4)] = 1
    test_y2[(test_y2 == 0) | (test_y2 == 4)] == 1
    valid_y2[(valid_y2 ==0) | (valid_y2 == 4)] == 1
    print(train_x.shape, train_y2.shape, test_x.shape, test_y2.shape, valid_x.shape, valid_y2.shape)
    model2 = Sequential()
    #model2.add(LSTM(64,input_shape=(1,90)))
    model2.add(Dense(60,input(90),activation='relu'))
    model2.add(Dense(1,activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit(train_x, train_y2, epochs=10, validation_data=(valid_x, valid_y2), shuffle=False,
              verbose=2, batch_size=5)
    # 模型评估
    scores2, results2 = model2.evaluate(test_x, test_y2, verbose=0)
    print('Accurcy2= %0.3f' % results2)

    pdb.set_trace()

    df_temp = precisionCalculate(pred_y,test_y_alter,test_start_date)
    df_result = pd.concat([df_result,df_temp],axis=0)
    print(df_result)

    print('Test result for %s ' % test_start_date)
    #将预测值，预测值概率和真实值生成csv文件
    def toCsv(pred_y,pred_y_prob, test_y):
        pred_y_prob = pd.DataFrame(pred_y_prob,columns=['up','hori','down'])
        prediction = pd.DataFrame(pred_y,columns=['result'])
        actual = pd.DataFrame(test_y, columns=['actual'])
        prediction = pd.concat([prediction,actual,pred_y_prob],axis=1)
        prediction['Date'] =ddate
        prediction['Time'] = ttime
        prediction.columns = ['Result','Actual','up','hori','down','Date','Time']
        prediction = prediction[['Date','Time','up','hori','down','Result','Actual']]
        prediction['Result'] = (prediction['Result'] -1) * (-1)
        prediction['Actual'] = (prediction['Actual']-1) * (-1)
        prediction.to_csv(operate_path + '/%s.csv' %test_start_date,index=False)
    toCsv(pred_y,pred_y_prob,test_y_alter)
    print('CSV has already been uploaded on date: %s' %test_start_date)


df_result.to_excel(operate_path +'/样本比例和准确率2.xlsx',index=False)






