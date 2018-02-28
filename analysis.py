from main import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.api import qqplot
loadDF()
df2 = pd.DataFrame()

main_factors = {'CA': 'TETGR TPOPP CLTCD TE PATCD HYTCB', 'AZ': 'GDPRX TPOPP TETGR TE HYTCB', 'NM': 'CLPRB NGTCD PATCD TE', 'TX': 'TE GDPRX CLTCD PATCD ESTCD TPOPP'}

# 处理一下主要因素的格式
for k in main_factors.keys():
    main_factors[k] = main_factors[k].split()

def normalize(arr):
    # return (np.ones(len(arr)) * np.max(arr) - arr) / (np.max(arr) - np.min(arr))
    # nonzero = np.nonzeros(arr)
    # if len(nonzero[0]) == 0:
        # return
    if np.std(arr) == 0:
        return arr
    else:
        return (arr - np.mean(arr)) / np.std(arr)
def getMSNMeaning():
    # data = pd.read_excel('data/partA_code_used.xlsx')
    # msn = data['MSN']
    # df2['MSN'] = msn
    d = []
    u = []
    msn = GDP_index
    # df2['Description'] = np.zeros(len('MSN'))
    for i in msn:
        df = msncodes.query("MSN=='%s'" % i)
        if df.size>0:
            d.append(df['Description'].values[0])
            u.append(df['Unit'].values[0])
        # print(df['Description'].values, df['Unit'].values)
    print(d, u)
    # df2['Description'] = d
    # df2['Unit'] = u

def getcode(k, code):
    if k[-1] == 'B':
        df = TotalDF[k]
        # print(df.columns)
        P = df[code]
        return P
def getCode(code):
    plt.figure()
    year = np.arange(1960, 2010)
    for k in TotalDF.keys():
        if k[-1] == 'B':
            df = TotalDF[k]
            # print(df.columns)
            P = df[code]
            plt.plot(year, P, label=k[:2])
            # print(k)
            # print(np.where(GDP==0))
    plt.legend()
    plt.show()
def plotCE(code): # 弹性系数
    CE = np.zeros(50)
    year = np.arange(1960, 2010)
    for k in TotalDF.keys():
        if k[-1]=='B':
            df = TotalDF[k]
            GDP = df['GDPRX']
            E = df[code]
            for i in range(50-1):
                if GDP[i] > 0:
                    CE[i] = (E[i+1]-E[i]) / (GDP[i+1]-GDP[i])

            plt.figure()
            plt.plot(year, CE)
            plt.title(k[:2])
            plt.show()
def linear(k, y, x):
    if len(x)==1:
        
        x = x[0].split()
        # print(x)
        x = [i.upper() for i in x]
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    df = TotalDF[k]
    if k[-1] == 'B':
        y = df[y]
        # indices = list(set(indices) - set(['CLPRB', 'HYTCB', 'NGMPB']))
        X = np.matrix([normalize(df[i].values) for i in x]).T
        X_r = X
        model = LinearRegression()
        model.fit(X_r, y)
        predictions = model.predict(X_r)
        # for i, prediction in enumerate(predictions):
        #     print('Predicted: %s, Target: %s' % (prediction, y[i]))
        year = np.arange(1960, 2010)
        ax.plot(year, predictions, '-', c='r', label='estimated', alpha=0.7)
        ax.plot(year, y, '-', c='g', label='real', alpha=0.7)
        ax.set_title(k[:2])
        ax.legend(loc=2)
        print('R-squared: %.2f ' % model.score(X_r, y))
        for index, coef in zip(x, model.coef_):
            print(index, coef)
    plt.show()
            # for index, coef in zip(indices, model.coef_):
            #     print(index, coef)

def pcaX(code):
    for k in TotalDF.keys():
        if k[-1]=='B':
            df = TotalDF[k]
            df_xr = pd.DataFrame()
            X_r = np.load('data/%s_%s_5main.npy' % (k, code))
            plt.figure()
            for i in range(5):
                df_xr[i] = X_r[:, i]
                plt.plot(year, X_r[:,i])
            plt.title(k)
            df_xr.to_csv('data/%s_%s_5main.csv' % (k, code))
            plt.show()


def plotDiff():
    x = getcode('CA_B', 'GDPRX')
    x = pd.Series(x)
    xdiff1 = x.diff()
    xdiff2 = x.diff(periods=2)
    xdiff3 = x.diff(periods=3)
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})

    ax1 = plt.subplot(221+0)
    ax1.plot(year[20:], x[20:])
    ax1.set_title('GDP')
    ax1.set_xticklabels('')

    ax2= plt.subplot(221+1)
    ax2.plot(year[20:], xdiff1[20:])
    ax2.set_title('first order of GDP')
    ax2.set_xticklabels('')
    ax2.set_yticklabels('')

    ax3=plt.subplot(221+2)
    ax3.plot(year[20:], xdiff2[20:])
    ax3.set_title('second order of GDP')

    ax4=plt.subplot(221+3)
    ax4.set_title('third order of GDP')
    ax4.plot(year[20:], xdiff3[20:])
    ax4.set_yticklabels('')

    plt.xlabel('year')
    plt.savefig('fig/gdp_diff.pdf')

    plt.show()


def betterPrecFile():
    for state in ['CA', 'AZ', 'NM', 'TX']:
    # for state in ['CA']:
        df0 = pd.DataFrame()
        
        for factor in main_factors[state]:
            ser = []
            if state!='AZ' or factor!='TETGR':
                df = pd.read_excel('data/%s_prec.xlsx' % state, sheetname=factor, header=None)
                for j in range(len(df)):
                    ser.append(float(df[0][j].split()[1]))
                # print(len(ser))
                # print(ser)
                df0[factor] = ser
        df0.to_excel('data/%s_betterprec.xlsx' % state)
                
            # print(df)

def predict():
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    for k in ['CA', 'AZ', 'NM', 'TX']:
        df = TotalDF[k+'_B']
        DF_pre = DF_predict[k]
        for e in ['CL', 'NG', 'NU', 'PM', 'RE']:
            Year = np.arange(1960, 2060)
            ser = []
            print(k, e)
            y = df[e]
            ser += list(y)
            X = np.matrix([df[i].values for i in main_factors[k]]).T
            model = LinearRegression()
            model.fit(X, y)
            print(model.score(X, y))
            df_prec = pd.read_excel('data/%s_betterprec.xlsx' % k)
            X_prec = np.matrix([df_prec[i].values for i in main_factors[k]]).T
            
            y_prec = model.predict(X_prec)
            ser += list(y_prec)
            plt.figure()
            plt.plot(Year, ser)
            DF_pre[e] = ser
            DF_pre[e+'_real'] = list(y) + list(np.zeros(len(y_prec)))
            plt.plot(np.arange(1960, 2010), model.predict(X), c='r')
            plt.title('%s %s' % (k, e))
            plt.show()

def plotBar():
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 10})
    for k in ['CA', 'AZ', 'NM', 'TX']:
        df = TotalDF[k+'_B']
        for e in E2009.keys():
            E2009[e].append(df[e].values[-1])
    name_list = ['CL', 'NG', 'NU', 'PM', 'RE']
    colors = ['black', 'red', 'blue', 'orange', 'green']
    # for i, k in enumerate(name_list):
    plt.bar(range(4), E2009['CL'], label='CL', fc=colors[0], alpha=0.3)
    plt.bar(range(4), E2009['PM'], bottom=E2009['CL'], label='PM', fc=colors[1], alpha=0.3)
    plt.bar(range(4), E2009['NG'], bottom=np.array(E2009['CL'])+np.array(E2009['PM']), label='NG', fc=colors[2], alpha=0.3)
    plt.bar(range(4), E2009['NU'], bottom=np.array(E2009['CL'])+np.array(E2009['PM'])+np.array(E2009['NG']), label='NU', fc=colors[3], alpha=0.3)
    plt.bar(range(4), E2009['RE'], bottom=np.array(E2009['CL'])+np.array(E2009['PM'])+np.array(E2009['NG'])+np.array(E2009['NU']), label='RE', fc=colors[4], tick_label = ['CA', 'AZ', 'NM', 'TX'], alpha=0.3)
    # plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='girl', tick_label=name_list, fc='r')
    plt.legend()
    plt.ylabel('billion btu')
    plt.tight_layout()
    plt.savefig('fig/bar.pdf')
    plt.show()

# plotDiff()
E2009 = {}
for i in ['CL', 'NG', 'NU', 'PM', 'RE']:
    E2009[i] = []

plotBar()
DF_predict = {}
# for state in ['CA', 'AZ', 'NM', 'TX']:
#     DF_predict[state] = pd.DataFrame()
#
# predict()
#
# for state in DF_predict.keys():
#     DF_predict[state].to_excel('data/%s_Prediction.xlsx' % state)
# x = getcode('CA_B', 'GDPRX').values
# x = x[20:]
# time = pd.date_range(start='1979', periods=30, freq='a')
# x = pd.Series(x, time)
# fig = plt.figure(figsize=(12, 9))
# plt.rcParams.update({'font.size': 18})
# ax1 = fig.add_subplot(211)
# dta = np.nan_to_num(x.diff())
# # dta = x.diff()
# fig = sm.graphics.tsa.plot_acf(dta, lags=20, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta, lags=20, ax=ax2)
# arma_mod80 = sm.tsa.ARMA(dta,(8,0)).fit()
# predict_dta = arma_mod80.predict('2090', '2100', dynamic=True)
# plt.savefig('fig/gdp_diff1_acf_pcf.pdf')

# fig, ax = plt.subplots()
# getMSNMeaning()
# df2.to_csv('data/partA_code.csv')
# getCode('TEEIB')
# linear('CA_B', 'Entropy', ['CLPRB', 'TPOPP', 'Temperature'])
# linear('AZ_B', 'Entropy', ['HYTCB', 'TERCB', 'TEEIB'])
# linear('NM_B', 'Entropy', ['TETGR', 'TEEIB', 'TEICB'])
# linear('TX_B', 'Entropy', ['GDPRX', 'TPOPP', 'NGMPB'])




# linear('CA_B', 'CL', ['gdprx tetgr cltcd te ngmpb patcd hytcb'])
# linear('CA_B', 'CL', ['gdprx tetgr teccb tercb tpopp'])
# linear('CA_B', 'PM', ['cltcd tpopp teacb teicb'])
# linear('CA_B', 'PM', ['cltcd tpopp te'])
# regress ng gdprx tetgr cltcd te patcd hytcb
# linear('CA_B', 'NG', ['gdprx tetgr cltcd te patcd hytcb'])
# linear('CA_B', 'NG', ['ngmpb cltcd patcd te'])
# linear('CA_B', 'NU', ['ngmpb tpopp teacb teccb'])
# linear('CA_B', 'NU', ['ngmpb tpopp te'])
# linear('CA_B', 'RE', ['hytcb gdprx te tpopp'])
# regress ng ngmpb cltcd ngtcd patcd te
# plotCE('CL')
# pcaX('CL')