from main import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
loadDF()

indices = []
for i in Index:
    indices += i
# def pacAnalysis():
def normalize(arr):
    # return (np.ones(len(arr)) * np.max(arr) - arr) / (np.max(arr) - np.min(arr))
    # nonzero = np.nonzeros(arr)
    # if len(nonzero[0]) == 0:
        # return
    if np.std(arr) == 0:
        return arr
    else:
        return (arr - np.mean(arr)) / np.std(arr)

def saveDF_norm():
    for k in TotalDF.keys():
        df = TotalDF[k]
        if k[-1]=='B':
            for i in indices:
                df[i] = normalize(df[i].values)
            df.to_csv('data/%s_norm.csv' % k)
def pac(code):
    for df_name in TotalDF.keys():
        df = TotalDF[df_name]
        # print(df_name)
        if df_name[-1] == 'B':
            y = df[code]
            indices = []
            for i in Index:
                indices += i
            # indices = list(set(indices) - set(['CLPRB', 'HYTCB', 'NGMPB']))
            X = np.matrix([normalize(df[i].values) for i in indices]).T
            # pca = PCA(n_components=len(indices))
            pca = PCA(n_components=5)
            # var_ratio = pca.explained_variance_ratio_
            s = 0
            # for i in range(len(pca.explained_variance_ratio_)):
            fit = pca.fit(X)
            # for i in range(len(indices)):
                # print(indices[i], pca.explained_variance_ratio_[i])
            print(pca.explained_variance_ratio_)
            X_r = fit.transform(X)
            np.save('data/%s_%s_5main' % (df_name, code), X_r)
            # print(X_r)
            # print(pca.explained_variance_)
            model = LinearRegression()
            model.fit(X_r, y)
            predictions = model.predict(X_r)
            # for i, prediction in enumerate(predictions):
            #     print('Predicted: %s, Target: %s' % (prediction, y[i]))
            fig, ax = plt.subplots()
            year = np.arange(1960, 2010)
            ax.plot(year, predictions, '-', c='r', label='prediction')
            ax.plot(year, y, '-', c='g', label='real')
            plt.title(df_name)
            plt.legend()
            plt.show()
            np.save('data/%s_%s_coef' % (df_name, code), model.coef_)
            print('R-squared: %.2f ' % model.score(X_r, y))
            # for index, coef in zip(indices, model.coef_):
                # print(index, coef)
def linear():
    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 6})
    count = 0
    for df_name in TotalDF.keys():
        df = TotalDF[df_name]
        # print(df_name)
        if df_name[-1] == 'B':
            y = df['Entropy']
            indices = ['CLPRB', 'TPOPP']

            # indices = list(set(indices) - set(['CLPRB', 'HYTCB', 'NGMPB']))
            X = np.matrix([normalize(df[i].values) for i in indices]).T
            X_r = X
            model = LinearRegression()
            model.fit(X_r, y)
            predictions = model.predict(X_r)
            # for i, prediction in enumerate(predictions):
            #     print('Predicted: %s, Target: %s' % (prediction, y[i]))
            year = np.arange(1960, 2010)
            ax = plt.subplot(221+count)
            ax.plot(year, predictions, '-', c='r', label='estimated', alpha=0.7)
            ax.plot(year, y, '-', c='g', label='real', alpha=0.7)
            ax.set_title(df_name[:2])
            ax.legend(loc=2)
            if count==0 or count==1:
                ax.set_xticklabels([])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            if count==1 or count==3:
                ax.set_yticklabels([])
            ax.set_ylabel('billion btu')
            ax.annotate('$R^2$=%.3f' % model.score(X_r, y), xy=(0.05, 0.5), xycoords='axes fraction')
            # ax.text(1990, 0.44, '$R^2$=%.3f' % model.score(X_r, y))
            for index, coef in zip(indices, model.coef_):
                print(index, coef)
            count+=1
    # plt.savefig('fig/RE_estimation.pdf')
    plt.show()
            # print('R-squared: %.2f ' % model.score(X_r, y))
            # for index, coef in zip(indices, model.coef_):
            #     print(index, coef)

def poly():
    for df_name in TotalDF.keys():
        df = TotalDF[df_name]
        y = df['Entropy']
        if df_name == 'AZ_B':
            indices = []
            for i in Index:
                indices += i
            # indices = list(set(indices) - set(['CLPRB', 'HYTCB', 'NGMPB']))
            X = np.matrix([normalize(df[i].values) for i in indices]).T
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly,y)

    print(lin_reg_2.intercept_)

    X_grid = np.arange(min(X),max(X),0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X,y,color = 'red')
    plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
    plt.title('predict(2-Polynomial Regression)')
    plt.xlabel('time')
    plt.ylabel('people count')
    plt.show()

def logistic():
    pass

# sc = StandardScaler()
# X_std = sc.transform(X)
# lr = LogisticRegression(C=1000.0, random_state=0)
# saveDF_norm()
# linear()
for code in ['CL', 'PM', 'NG', 'NU', 'RE']:
    pac(code)
# poly()
# linear()