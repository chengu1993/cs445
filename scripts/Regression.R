setwd("E:/Study/Yale/CPSC 545/project/NBA")

data<-read.csv("shot_logs.csv")

################################################
head(data,30)
data$month<-substr(data[,2],1,3)


col<-c(3,7,8,9,10,11,12,13,16,17,18,20,21)
  
regdata<- data[,col]
regdata[is.na(regdata[,4]),4]<-0

regdata[,3]<-(as.numeric(as.POSIXct(strptime(regdata[,3], format = "%M:%S"))) - 
    as.numeric(as.POSIXct(strptime("0", format = "%S"))))

regdata[,2]<-as.factor(regdata[,2])
regdata[,8]<-as.factor(regdata[,8])
regdata[,8]<-as.factor(regdata[,8])
#regdata<-regdata[regdata$TOUCH_TIME>=0,]

feed<-regdata[,c(11,1:8,10)]
  
library(YaleToolkit)
whatis(feed)
head(feed,30)

model <- glm(FGM ~.,family=binomial(link='logit'),data=feed)
summary(model)
library("car")
vif(model)


model2 <- glm(FGM ~.,family=binomial(link='logit'),data=feed[,c(-4,-7,-9)])
summary(model2)
vif(model2)


#########################################
library(dplyr)
###Location:
feed %>%
  group_by(LOCATION) %>%
  summarise(mean(FGM))

###Period:
feed %>%
  group_by(PERIOD) %>%
  summarise(mean(FGM))

###Shot Clock:
Rshotc<- feed %>%
  group_by(round(SHOT_CLOCK)) %>%
  summarise(mean(FGM))


Mshotc <- glm(FGM ~ SHOT_CLOCK,family=binomial(link='logit'), data=feed)
summary(Mshotc)
plot(Rshotc)
x<-seq(0, 25, 0.01)
y <- predict(Mshotc, data.frame(SHOT_CLOCK = x ) ,type = "response")
lines(x,y)

###Dribbles:

Rdrib<- feed %>%
  group_by(DRIBBLES) %>%
  summarise(mean(FGM))


Mdrib <- glm(FGM ~ DRIBBLES,family=binomial(link='logit'), data=feed)
plot(Rdrib)
x<-seq(0, 35, 0.01)
y <- predict(Mdrib, data.frame(DRIBBLES = x ) ,type = "response")
lines(x,y)

###Shot Distance:
Rshotd<- feed %>%
  group_by(round(SHOT_DIST)) %>%
  summarise(mean(FGM))


Mshotd <- glm(FGM ~ SHOT_DIST,family=binomial(link='logit'), data=feed)
plot(Rshotd)
x<-seq(0, 50, 0.01)
y <- predict(Mshotd, data.frame(SHOT_DIST = x ) ,type = "response")
lines(x,y)

### Points Type:
feed %>%
  group_by(PTS_TYPE) %>%
  summarise(mean(FGM))


Mpts <- glm(FGM ~PTS_TYPE,family=binomial(link='logit'),data=feed)
summary(Mpts)
###Closest Defender distance
Rcdd<- feed %>%
  group_by(round(CLOSE_DEF_DIST)) %>%
  summarise(mean(FGM))


Mcdd <- glm(FGM ~ CLOSE_DEF_DIST,family=binomial(link='logit'), data=feed)
plot(Rcdd)
x<-seq(0, 60, 0.01)
y <- predict(Mcdd, data.frame(CLOSE_DEF_DIST = x ) ,type = "response")
lines(x,y)
