from main import *
# import numpy.linalg.solve as solve
loadDF()

As = []
p2009 = []
def test(year, df):
    # print(year)
    n = 5
    CL = df['CL'].values
    NG = df['NG'].values
    NU = df['NU'].values
    RE = df['RE'].values
    PM = df['PM'].values
    TE = CL + NG+NU+RE+PM
    # EI = TE-PM-RE-NU-NG-CL
    Ser1 = (CL / TE)
    Ser2 = (NG / TE)
    Ser3 = (NU / TE)
    Ser4 = (RE / TE)
    Ser5 = (PM / TE)
    # Ser6 = (EI / TE)[year:year+length+1]
    Sers = [Ser1, Ser2, Ser3, Ser4, Ser5]
    for i in Sers:
        if len(p2009)<5:
            p2009.append(i[-1])
       # print(i[-1])
    A = np.zeros([n, n])
    # 计算保留概率
    for i in range(n):
        # 检查消费份额增加或不变的能源
        seri = Sers[i]
        if seri[year+1] >= seri[year]:
            # print(i)
            A[i][i] = 1
            # 计算保留概率为1的元素所在行的转移概率
            # 已经为0了
        else:
            # 该列的吸收概率概率为0
            # 确定该元素所在行的非零转移概率
            down = 0
            for j in range(n):
                if j != i:
                    serj = Sers[j]
                    if serj[year + 1] > serj[year]:
                        down += serj[year + 1] - serj[year]
                    # if serj[year + 1] > serj[year]:
                    #     print(i, j, up, down)
                    #     A[i][j] = up / down
            for j in range(n):
                if j != i:
                    serj = Sers[j]
                    up = (1 - A[i][i]) * (serj[year + 1] - serj[year])
                    if serj[year + 1] > serj[year]:
                        # print(i, j, up, down)
                        A[i][j] = up / down
    # print(A)
    # print(len(Ser1))
    # rs = [Ser1, Ser2, Ser3, Ser4, Ser5, Ser6]
    # # A = np.matrix([Ser1[:length-1], Ser2[:length-1], Ser3[:length-1], Ser4[:length-1], Ser5[:length-1], Ser6[:length-1]]).T
    # A = np.matrix([Ser1[:-1], Ser2[:-1], Ser3[:-1], Ser4[:-1], Ser5[:-1], Ser6[:-1]]).T
    # print(A.shape)
    
    # for i, r in enumerate(rs):
    #     print(i)
    #     right = r[1:]
    #     print(np.linalg.solve(A, right))
    # #print(Ser1)
    As.append(A)
    # print(A)
    return A
for y in range(40, 49):
    test(y, TotalDF['CA_B'])

A0 = As[0]
for i in range(1, len(As)):
    # print(A0)
    A0 += As[i]
A0 /= len(As)
# p2009 = np.array([0.238193824372,0.217106968363,0.184857578937,0.0596512269837,0.300190401344])

def prec():
    fig, ax = plt.subplots()
    P = [p2009]
    p = p2009
    for i in range(40):
        p = np.dot(A0, p)
        P.append(p)
    year = np.arange(2009, 2009+len(P))
    colors = ['blue', 'black', 'red', 'yellow', 'green']
    for i in range(5):
        xi = [sum(p[:i+1]) for p in P]
        # scale = sum()
        ax.plot(year, xi, color=colors[i])
        # print(p)
    plt.show()
prec()
def fujian(year):
    df = pd.read_excel('data/fujian.xlsx')
    n = 4
    A = []
    Sers = []
    for i in range(n):
        ar = []
        for j in range(8):
            strf = str(df[2004+j][i]).replace(' ','')
            # print(strf.strip())
            ar.append(float(strf)/ 100)
        A.append(ar)
        Sers.append(ar)
        # print(ar)
    # A = np.matrix(A).T
    A = np.zeros([n, n])
        # 计算保留概率
    for i in range(n):
        # 检查消费份额增加或不变的能源
        seri = Sers[i]
        if seri[year+1] >= seri[year]:
            A[i][i] = 1
            # 计算保留概率为1的元素所在行的转移概率
            # 已经为0了
        else:
            # 该列的吸收概率概率为0
            # 确定该元素所在行的非零转移概率
            A[i][i] = seri[year+1] / seri[year]
            # print(i, seri[year+1], seri[year])
            down = 0
            for j in range(n):
                if j != i:
                    serj = Sers[j]
                    if serj[year+1]>serj[year]:
                        down += serj[year+1]-serj[year]
            for j in range(n):
                if j != i:
                    serj = Sers[j]
                    up = (1-A[i][i]) * (serj[year+1]-serj[year])
                    if serj[year+1]>serj[year]:
                        # print(i, j, up, down)
                        A[i][j] += up/down
    # print(A)
    return np.matrix(A)


# As = []
# for year in range(7):
#     As.append(fujian(year))
#
# A0 = As[0]
# for i in range(1, 7):
#     # A0 = np.dot(A0, As[i])
#     A0 += As[i]
#     print(A0)
# A0 = A0 / 7

# for i in range(n):
    # print(sum(A[i][:]))
    # if A[i][i] < 1:
        # 该列需归零

#     # 检查消费份额增加或不变的能源
#     seri = Sers[i]
#     if seri[year+1] >= seri[year]:
#         A[i][i] = 1
#     else:
#         for j in range(n):
#             if j!=i:
#                 A[j][i]=0
# print(A)

















# for i in range(n):
#     ar = []
#     for j in range(n):
#         strf = str(df[2004+j+1][i]).replace(' ','')
#         # print(strf.strip())
#         ar.append(float(strf)/100)
#     print(ar)
#     p = np.linalg.solve(A, ar)
#     print(sum(p))
# All = np.array([[75, 17.5, 7.5], [64.5, 24, 11.5], [55.5, 29.2, 15.3], [47.7, 33.2, 19.1]]) / 100
# A = All[:3,:]
# row2 = np.array([24, 29.2, 33.2]) / 100
# p1 = np.linalg.solve(A, row2)
# # print(np.linalg.solve(A, row2))

# print(np.dot(A, p1))
# p1_ans =  np.array([0.86, 0, 0])
# p1_ans =  np.array([0.098, 0.94, 0.025])
# print(np.dot(A, p1_ans))
# Ser1 = np.array([75, 64.5, 55.5, 47.7]) / 100
# Ser2 = np.array([17.5, 24, 29.2, 33.2]) / 100
# Ser3 = np.array([7.5, 11.5, 15.3, 19.1]) / 100
# rs = [Ser1, Ser2, Ser3]
# A = np.matrix([Ser1[:-1], Ser2[:-1], Ser3[:-1]]).T
# for i in rs:
#     right = i[1:]
#     print(np.linalg.solve(A, right))