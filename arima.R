# dev.new()
setwd("C://Users//11316//Documents//0MCM//data")
library(forecast)
data = read.table("CA_B.csv",header=T, sep=",")
# X <- data$CLTCD
# Xts <- ts(X, start=c(1960))
X<-data$GDPRX[(20:48)]
Xts <- ts(X,start = c(1980))
plot(Xts)
acf(Xts)
Xtsdiff1<-diff(Xts, differences = 1)
# plot(Xtsdiff1)
adf.test(Xtsdiff1)
Xtsdiff2<-diff(Xts, differences = 2)
adf.test(Xtsdiff2)
Xtsdiff3<-diff(Xts, differences = 3)
adf.test(Xtsdiff3)

Xtsdiff4<-diff(Xts, differences = 4)
adf.test(Xtsdiff4)

plot.ts(Xtsdiff3)
plot.ts(Xtsdiff2)
acf(Xtsdiff1, lag.max = 20)
pacf(Xtsdiff1,lag.max=20)

X<-Xtsdiff3
acf(X, lag.max = 20)
pacf(X,lag.max=20)

acf(Xtsdiff3)
pacf(Xtsdiff3)

Xarima <-arima(Xts, order=c(2,3,1))
Xarimaforecast <- forecast(Xarima, h=50, level=c(99.5))
plot(Xarimaforecast)
