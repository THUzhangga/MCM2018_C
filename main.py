import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt

# global CA_B,  CA_V, AZ_B, AZ_V, NM_B, NM_V,TX_B, TX_V #定义全局变量
# 导入数据
seseds = pd.read_csv('data/seseds.csv')
msncodes = pd.read_csv('data/msncodes.csv')

# 取出seseds各列
data = seseds['Data'].values
msn = seseds['MSN'].values

statecode = seseds['StateCode'].values

# 取出msncode各列
unit = msncodes['Unit'].values
MSN = msncodes['MSN'].values #用大写区分前面的msn列
description = msncodes['Description'].values #用大写区分前面的msn列

year = np.arange(1960, 2010)
def getSer(code, sc): #获取一个msncode和一个sc对应的时间序列
    ser = np.zeros(50)
    df = seseds.query("MSN=='%s' and StateCode=='%s'" % (code, sc))
    for y in df['Year'].values:
        ser[int(y) - 1960] = df.query('Year==%d' % int(y))['Data'].values  # 时间对上
    return ser

def getSerForDF(code_list):
    for code in code_list:
        for k in TotalDF.keys():
            df = TotalDF[k]
            df[code] = getSer(code, k[:2])
def getCategory(df_sc, code_cat, sc, last='B'):
    '''
    获取一类的时间序列
    :param code_cat: key为类名，value为某一类包含的前两个字母
    :param sc: startcode，州码
    :param last: 最后一位字母
    :return: 时间序列
    '''
    ser = np.zeros(50) # 50年的时间序列
    for xx in code_cat: # xx是两个字母的字符串，即msncode的前两位
        if xx != 'NU':
            code = xx + 'TC' + last
        elif xx == 'NU': # 对于核能，部门有所不同
            code = xx + 'ET' + last
        df_sc[xx] = getSer(code, sc)

def getDF(code_cat):
    for df_cat_name in TotalDF.keys():
        df = TotalDF[df_cat_name]
        getCategory(df, code_cat, df_cat_name[0:2], df_cat_name[-1])
        ser = np.zeros(50)
        for i in CAT['RE']:
            ser += df[i]
        df['RE'] = ser # 汇总可再生能源
        df['PM'] = df['PA'].values - df['EM'].values # 注意PM

def getIndex():
    for df_cat_name in TotalDF.keys():
        df = TotalDF[df_cat_name]
        sc = df_cat_name[0:2]
        for index in Index: # index为某一类指标
            for i in index: # i为某一指标
                if i != 'RETCD': # RETCD暂且没有
                    df[i] = getSer(i, sc)

def saveDF(): #将DF保存到文件中
    for k in TotalDF.keys():
        TotalDF[k].to_csv('data/%s.csv' % k)

def loadDF(): #读入DF
    for name in nameDF:
        TotalDF[name] = pd.read_csv('data/%s.csv' % name)

def plotDF():
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    year = np.arange(1960, 2010)
    for k in TotalDF.keys():
        if k[-1] == 'B':
            df = TotalDF[k]
            TE = np.zeros(50)
            for cat in CAT.keys():
                if cat != 'TE':
                    TE += df[cat]
            ax.plot(year, TE, label = k[0:2], alpha=0.8)
            # ax.plot(year, df[cat], label = '')
    ax.set_xlabel('year')
    ylabel = 'billion btu'
    ax.set_ylabel(ylabel)
    plt.legend()
    # plt.title('Amount of energy comsumed by the four states from 1960 to 2009')
    plt.savefig('fig/amount.pdf')
    plt.show()

def getEntropy():
    for k in TotalDF.keys():
        if k[-1] == 'B': # 只考虑消费的
            df = TotalDF[k]
            # 计算每年各类能量总和
            s = df['TE']
            # 计算各类能源比例
            p = np.zeros(50)
            for c in TypesEnergy:
                if c != 'TE':
                    df[c+'_prop'] = df[c] / s
            # 计算信息熵
            S = np.zeros(50)
            for i in range(50):
                for c in TypesEnergy:
                    P_i = df[c + '_prop'][i]
                    if P_i > 0: # 比例大于0的才计算
                        S[i] += -P_i * np.log10(P_i)
            # 计算均衡度
            E = np.zeros(50)
            for i in range(50):
                prop = []
                for c in TypesEnergy:
                    P_i = df[c + '_prop'][i]
                    if P_i > 0: # 比例大于0的才计算
                        prop.append(P_i)
                n = len(prop) #能源种类数
                E[i] = S[i] / np.log10(n)
            df['Entropy'] = S
            df['BalanceDegree'] = E
            df['DominanceDegree'] = 1- E

def addWeather():
    prec = pd.read_excel('data/weather.xlsx', sheetname='precipitation')
    t = pd.read_excel('data/weather.xlsx', sheetname='temperature')
    for df_name in TotalDF.keys():
        df = TotalDF[df_name]
        sc = df_name[0:2]
        df['Precipitation'] = prec[sc].values
        df['Temperature'] = t[sc].values

def getTE():
    for k in TotalDF.keys():
        df = TotalDF[k]
        s = np.zeros(50)
        for i in ['CL', 'NG', 'NU', 'PM', 'RE', 'ELISB']:
            s += df[i]
        df['TE'] = s

def getRETCD():
    # 首先需要获得总的花费
    for k in TotalDF.keys():
        df = TotalDF[k]
        df['TETCV'] = getSer('TETCV', sc=k[0:2])
        s = np.zeros(50)
        for i in Price_index:
            if i != 'RETCD':
                s += df[i[0:2]] * df[i] # 对应的量乘以对应的价格（除了RE外）

        # 此外再考虑电的进出口
        df['ESTCD'] = getSer('ESTCD', k[:2])
        s += df['ESTCD'] * df['ELISB']
        df['RETCD'] = (df['TETCV'] - s) / df['RE']

def getGrowthRate():
    for k in TotalDF.keys():
        df = TotalDF[k]
        GDP = df['GDPRX']
        s = np.zeros(50)
        for i in range(17, 49):
            if GDP[i] > 0:
                s[i] = (GDP[i+1] - GDP[i]) / GDP[i]
        s[-1] = s[-2]
        df['GrowthRate'] = s
def plotEntropy():
    # fig, ax = plt.subplots()
    i = 0
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 8})
    for k in TotalDF.keys():
        if k[-1] == 'B':
            
            S = TotalDF[k]['Entropy']
            E = TotalDF[k]['BalanceDegree']
            D = TotalDF[k]['DominanceDegree']
            year = np.arange(1960, 2010)
            axi = plt.subplot(221+i)
            axi.plot(year, S, label='$S$')
            axi.plot(year, E, label='$E$')
            axi.plot(year, D, label='$D$')
            axi.set_xlabel('year')
            if i == 0 or i == 2:
                axi.set_ylabel('entropy or degree [-]')
            axi.set_ylim([0, 1])
            axi.legend(loc=2)
            axi.set_title(k[0:2])
            i+=1
    plt.tight_layout()
    plt.savefig('fig/entropy.pdf')
    plt.show()

def plotProfile():
    # fig, ax = plt.subplots()
    i = 0
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    for k in TotalDF.keys():
        if k[-1] == 'B':
            axi = plt.subplot(221 + i)
            df = TotalDF[k]
            CL = df['CL'].values
            PM = df['PM'].values
            NG = df['NG'].values
            NU = df['NU'].values
            RE = df['RE'].values
            EL = df['ELISB'].values
            year = np.arange(1960, 2010)
            S = CL + PM + NG + NU + RE + EL
            color = ['grey', 'red', 'blue', 'orange', 'green']
            lw = 0.5
            axi.plot(year, CL/S, lw=lw, color='black')
            axi.fill_between(year, CL/S, color='black', alpha=0.3)

            axi.plot(year, (CL+PM)/S, lw=lw, color='black')
            axi.fill_between(year, CL / S, (CL+PM)/S, facecolor='red', alpha=0.3)

            axi.plot(year, (CL+PM+NG)/S, lw=lw, color='black')
            axi.fill_between(year, (CL + PM)/S, (CL+PM+NG)/S, facecolor='blue', alpha=0.3)

            axi.plot(year, (CL+PM+NG+NU)/S, lw=lw, color='black')
            axi.fill_between(year, (CL+PM+NG)/S, (CL+PM+NG+NU)/S, facecolor='orange', alpha=0.3)

            axi.plot(year, (CL + PM + NG + NU + RE) / S, lw=lw, color='black')
            axi.fill_between(year, (CL+PM+NG+NU) / S, (CL + PM + NG + NU + RE) / S, facecolor='green', alpha=0.3)

            axi.plot(year, np.ones(50), lw=lw*2, color='black')

            # axi.fill_between(year, (CL + PM + NG + NU + RE) / S, np.ones(50), facecolors='green', alpha=0.3)
            # if len(np.where(EL<0)) > 0:
            #     pass
            #     # axi.fill_between(year, (CL+PM+NG+NU+EL)/S, np.ones(50), facecolors='green', alpha=0.3)
            # else:
            #     print(k)
            #     axi.fill_between(year, (CL + PM + NG + NU + EL) / S, np.ones(50), facecolors='green', alpha=0.3)
            if i==0 or i==1:
                axi.set_xticklabels([])
            if i==2 or i==3:
                axi.set_xlabel('year')
            if i == 0 or i == 2:
                
                axi.set_ylabel('proportion [-]')
            # axi.set_ylim([0, 1])
            # axi.legend(loc=2)
            axi.set_title(k[0:2])
            i += 1
    # plt.legend(frameon=True, loc='NorthOutside')
    # plt.tight_layout()
    width = 0.12
    plt.text(2012, 0, 'CL', bbox={'facecolor': 'black', 'alpha': 0.3, 'pad': 1})
    plt.text(2012, width, 'PM', bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 1})
    plt.text(2012, width*2, 'NG', bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 1})
    plt.text(2012, width*3, 'NU', bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 1})
    # plt.text(2012, width*4, 'EL', bbox={'facecolor': 'pink', 'alpha': 0.3, 'pad': 1})
    plt.text(2012, width*4, 'RE', bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 1})
    plt.savefig('fig/proportion.pdf')
    plt.show()
# TotalDF = {'CA_B': CA_B, 'CA_V': CA_V, 'AZ_B': AZ_B, 'AZ_V': AZ_V, 'NM_B': NM_B, 'NM_V': NM_V, 'TX_B': TX_B, 'TX_V': TX_V}
TotalDF = {}
nameDF = ['CA_B', 'CA_V', 'AZ_B', 'AZ_V', 'NM_B', 'NM_V', 'TX_B', 'TX_V']
for name in nameDF:
    TotalDF[name] = pd.DataFrame()
    # TotalDF[name].index = np.arange(50)
code_cat = ['CL', 'NG', 'PM', 'NU', 'EM', 'RE', 'PA']

Geo_index = ['CLPRB', 'HYTCB', 'NGMPB', 'PAPRB'] # 地理因素的指标
# Geo_index = []
GDP_index = ['GDPRX', 'TETGR', 'CLTCD', 'NGTCD', 'PATCD', 'NUETD', 'RETCD', 'ESTCD'] # 经济因素指标
Added_GDP_index = ['GrowthRate']
Price_index = ['CLTCD', 'NGTCD', 'PATCD', 'NUETD', 'RETCD']
Human_index = ['TPOPP'] # 人口因素指标
Industry_index = ['TEACB', 'TECCB', 'TEICB', 'TEEIB', 'TERCB'] #工业因素指标
Weather_index = ['Precipitation', 'Temperature']
Index = [Geo_index, GDP_index, Added_GDP_index, Human_index, Industry_index, Weather_index] # 全部指标

# 注意ww=wo + ws
CAT = {'CL': ['CL'], 'NG': ['NG'], 'PM': ['PA', 'EM'], 'NU': ['NU'], 'RE':['RE'], 'TE': ['TE']}
TypesEnergy = ['CL', 'NG', 'PM', 'NU', 'RE', 'ELISB']

if __name__ == '__main__':

    # loadDF()
    # plotEntropy()
    # plotProfile()
    # for k in TotalDF.keys():
    #     if k[-1] == 'B':
    #         re = TotalDF[k]['RE']
    #         year = np.arange(1960, 2010)
    #         plt.plot(year, re)
    #         plt.title(k)
    #         plt.show()
    #         # plotDF(TotalDF[k], k)
    # plotDF()

    # getDF(code_cat)
    # print('getDF')
    # getEntropy()
    # print('getEntropy')
    # getIndex()
    # print('getIndex')
    # getRETCD()
    # print('getRETCD')
    # # saveDF()
    # # loadDF()
    # addWeather()
    # saveDF()
    loadDF()
    # getRETCD()
    # getGrowthRate()
    # saveDF()
    plotProfile()
    # getSerForDF(['ELISB'])
    # getTE()
    # for k in TotalDF.keys():
    #     if k[-1] == 'B':
    #         df = TotalDF[k]
    #         df['TETCB'] = getSer('TETCB', k[:2])
    #         year = np.arange(1960, 2010)
    #         plt.plot(year, df['TETCB'], color = 'green', label='TETCB')
    #         plt.plot(year, df['TE'], color = 'red', label='CL+NG+NU+RE+PM')
    #         plt.title(k[:2])
    #         plt.legend()
    #         plt.show()
    # plotProfile()
    # getEntropy()
    # saveDF()
    # plotEntropy()

# for k in TotalDF.keys():
#     print(k)
#     plotDF(TotalDF[k], k)

# listoftype = []
# for i in msn:
#     listoftype.append(i[2:4])
# # listoftype = sorted(list(set(listoftype)))
# # print(listoftype)
#
# d = collections.Counter(listoftype)
# # 瞬间出结果
# listoftype = [i for i in d.keys()]
# listoftype.sort()
# count = 0
# for k in listoftype:
#     print(k, d[k])
#     count+= d[k]
#     # k是lst中的每个元素
#     # d[k]是k在lst中出现的次数


