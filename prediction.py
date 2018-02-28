from main import *
# loadDF()
def plotPredction():
    # fig, ax = plt.subplots()
    i = 0
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    for k in DF_predict.keys():
        axi = plt.subplot(221 + i)
        df = DF_predict[k]
        CL = df['CL'].values
        PM = df['PM'].values
        NG = df['NG'].values
        NU = df['NU'].values
        RE = df['RE'].values
        year = np.arange(1960, 2060)
        S = CL + PM + NG + NU + RE
        color = ['grey', 'red', 'blue', 'orange', 'green']
        lw = 0.5
        axi.plot(year, CL / S, lw=lw, color='black')
        axi.fill_between(year, CL / S, color='black', alpha=0.3)

        axi.plot(year, (CL + PM) / S, lw=lw, color='black')
        axi.fill_between(year, CL / S, (CL + PM) / S, facecolor='red', alpha=0.3)

        axi.plot(year, (CL + PM + NG) / S, lw=lw, color='black')
        axi.fill_between(year, (CL + PM) / S, (CL + PM + NG) / S, facecolor='blue', alpha=0.3)

        axi.plot(year, (CL + PM + NG + NU) / S, lw=lw, color='black')
        axi.fill_between(year, (CL + PM + NG) / S, (CL + PM + NG + NU) / S, facecolor='orange', alpha=0.3)

        axi.plot(year, (CL + PM + NG + NU + RE) / S, lw=lw, color='black')
        axi.fill_between(year, (CL + PM + NG + NU) / S, (CL + PM + NG + NU + RE) / S, facecolor='green', alpha=0.3)

        # axi.plot(year, np.ones(50), lw=lw, color='black')

        # axi.fill_between(year, (CL + PM + NG + NU + RE) / S, np.ones(50), facecolors='green', alpha=0.3)
        # if len(np.where(EL<0)) > 0:
        #     pass
        #     # axi.fill_between(year, (CL+PM+NG+NU+EL)/S, np.ones(50), facecolors='green', alpha=0.3)
        # else:
        #     print(k)
        #     axi.fill_between(year, (CL + PM + NG + NU + EL) / S, np.ones(50), facecolors='green', alpha=0.3)
        if i == 0 or i == 1:
            axi.set_xticklabels([])
        if i == 2 or i == 3:
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
    plt.text(2065, 0, 'CL', bbox={'facecolor': 'black', 'alpha': 0.3, 'pad': 1})
    plt.text(2065, width, 'PM', bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 1})
    plt.text(2065, width * 2, 'NG', bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 1})
    plt.text(2065, width * 3, 'NU', bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 1})
    # plt.text(2065, width*4, 'EL', bbox={'facecolor': 'pink', 'alpha': 0.3, 'pad': 1})
    plt.text(2065, width * 4, 'RE', bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 1})
    plt.savefig('fig/proportion_Prediction.pdf')
    plt.show()

if __name__ == '__main__':
    DF_predict = {}
    for state in ['CA', 'AZ', 'NM', 'TX']:
        DF_predict[state] = pd.read_excel('data/%s_Prediction.xlsx' % state)
        # DF_predict[state].index = pd.date_range(start=1960, periods=100, freq='a')
    plotPredction()