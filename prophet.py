from main import *
from fbprophet import Prophet
import datetime as dt
loadDF()

# import pandas as pd
import datetime
def datelist(beginyear, endyear):
    # beginyear, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.datetime(x, 1, 1) for x in range(beginyear, endyear)]
    return date_l

DF = TotalDF['AZ_B']
CL = DF['RE'].values

df = pd.DataFrame()
df['y'] = CL
df['ds'] = datelist(1960, 2010)
# df.set_index('year')

# df.index =
# df = pd.read_csv('data/fb.csv')
#df['y'] = np.log(df['y'])
# df.head()
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)