from main import *
loadDF()
s = []
def getSigma(x):
    n = len(x)
    s = []
    for i in range(1, len(x)):
        sigma = x[i-1] / x[i]
        s.append(sigma)
        
        if sigma >= np.exp(2/(n+1)) or sigma <= np.exp(-2/(n+1)):
            print(sigma,np.exp(2/(n+1)), np.exp(-2/(n+1)))
            return False
    # print(s)
    return True


def getAlpha_hat(x):
    n = len(x)
    x1 = np.cumsum(x)
    def getB(x1):
        ser = [-(x1[i]+x1[i+1])/2 for i in range(n-1)]
        return np.matrix([ser, np.ones(n-1)]).T
    B = getB(x1)
    Y = x[1:]
    alpha_hat = np.dot(np.dot(np.linalg.inv(B.T * B), B.T), Y)
    print(alpha_hat.reshape(-1, 2))
    return [alpha_hat[0,0], alpha_hat[0, 1]]

def getPre(x, alpha_hat):
    n = len(x)
    x1 = np.cumsum(x)
    a = alpha_hat[0]
    b = alpha_hat[1]
    x1_hat = np.zeros(n+2)
    for i in range(n+1):
        x1_hat[i+1] = (x1[0]-b/a)*np.exp(-a*i) + b/a 
    x0_hat = np.zeros(n)
    for i in range(n):
        x0_hat[i] = x1_hat[i+1] - x1_hat[i]
    return x0_hat

def plotPrc(code, hat=True):
    year = np.arange(1960, 2010)
    for k in TotalDF.keys():
        if k[-1] =='B':
            print(k)
            x = TotalDF[k][code].values[-10:]
            if getSigma(x) ==False:
                raise(ValueError)
            plt.figure()
            year = year[-10:]
            plt.plot(year, x, label='real')
            x_hat = getPre(x, getAlpha_hat(x))
            plt.plot(year, x_hat, label='hat')
            plt.title(k[:2])
            plt.show()
x = np.array([2.1398, 2.5105, 2.8250, 3.0596, 3.2225, 3.4536, 3.6357])
# getSigma(x)
# x = TotalDF['CA_B']['TE'].values
alpha = getAlpha_hat(x)
x_hat = getPre(x, alpha)
print(getPre(x, alpha))

plotPrc('TPOPP', True)