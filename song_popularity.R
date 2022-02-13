setwd(dir = "D:\\IIM - K\\SUPERVISED MACHINE LEARNING\\REGRESSION\\REG-PRACTICALS\\SongPopularity")

#Read the data
data<-read.csv("song_data.csv")

--------------------------------------------------------------------------------
#Step1: Data Pre-Processing and Exploratory Data Analysis

str(data)
table(data$song_name)

#we have removed the first column "Song name"

data<-data[,-c(1)]

#Checked for missing values and found there were no missing values

sum(is.na(data))

data.frame(colnames(data))

--------------------------------------------------------------------------------
  
#To understand the relationship between following categorical data and their impact on the output
#we shall conduct a chi-square test to examine the same. We assumed alpha as 0.05 and 
#and if p-values of these categorical data  is greater than alpha, 
#those variables will be removed as they are not significant. 
#Lesser the p-value highly significant it is to the model.

#Following are the categorical data in the dataset

#key
#audio_mode
#time_signature

chisq.test(data$key, data$song_popularity, simulate.p.value = TRUE)
chisq.test(data$audio_mode, data$song_popularity, simulate.p.value = TRUE)
chisq.test(data$time_signature, data$song_popularity, simulate.p.value = TRUE)

#From the chi-square test results, we found that the "time_signature" variable has high p-value i.e., 0.02>0.05
#hence the same shall be removed

data<-data[,-c(13)]

--------------------------------------------------------------------------------

#To identify the correlation between the independent variables (that are continuous in nature)
#we used pairs.panels test 

install.packages("psych")
library(psych)

str(data)
data.frame(colnames(data))

cor<-data[,c(2,3,4,5,6,8,9,11,12,13)]

pairs.panels(cor, scale = TRUE, digits = 2, method = "pearson")

#There is correlation between energy and loudness of about 0.76 which is moderately high 
#however we will not remove the variable at this stage. We will run Regression, check vif and then decide.

----------------------------------------------------------------------------------
  
#Step2: Split the data into Training & Test (validation)
  
install.packages("caTools")
library(caTools)

?sample.split
split<-sample.split(data, SplitRatio = 0.80)

train<-subset(data, split==TRUE)
test<-subset(data, split==FALSE)

--------------------------------------------------------------------------------

#Step3: Perform Linear Regression
  
install.packages("caret", dependencies = TRUE)
library(caret)

install.packages("car")
library(car)

?lm

set.seed(123)
linear<-lm(song_popularity~.,data = train)
summary(linear)
options(scipen = 10)  
coef(linear)

#From the results of the linear regression, we found that there the following variables are insignificant
#song_duration_ms, key, audio_mode, speechiness, tempo

data.frame(colnames(data))
data<-data[,-c(2,7,10,11,12)]
train<-train[,-c(2,7,10,11,12)]
test<-test[,-c(2,7,10,11,12)]


#Re-run Linear and step wise regression using the revised data

set.seed(123)
linear<-lm(song_popularity~.,data = train)
summary(linear)
options(scipen = 10)  
coef(linear)

vif(linear)

#vif for the predictors are less than 4, hence we will not exclude any of the variables at this stage

stepreg<-step(linear, direction = "backward", trace = 0)
summary(stepreg)

-------------------------------------------------------------------------------------
  
#Step4: Perform 10-fold cross validation

install.packages("glmnet")
library(glmnet)

custom<-trainControl(method = "repeatedcv", number = 10, repeats = 5)


?train
cv<-train(song_popularity~., data = train, method = "lm", trControl = custom)
summary(cv)

plot(varImp(cv, scale = TRUE))

--------------------------------------------------------------------------------
#Step5: Perform Ridge, Lasso, Elastic net Regression

#Ridge
  
set.seed(123)
ridge<-train(song_popularity~., data = train, method = "glmnet", 
             trControl = custom,
             tuneGrid = expand.grid(alpha = 0, lambda = seq(0.01, 1, length = 5)))
  
summary(ridge)  
ridge$results  
ridge$bestTune 

#alpha = 0, lambda = 0.2575, RMSE = 21.35

plot(varImp(ridge, scale = TRUE))  

---------------------------------------------------------------------------------

#lasso
  
set.seed(123)
lasso<-train(song_popularity~., data = train, method = "glmnet", 
             trControl = custom,
             tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 5)))

summary(lasso)  
lasso$results  
lasso$bestTune 

#alpha = 1, lambda = 0.00100, RMSE = 21.35

plot(varImp(lasso, scale = TRUE))    
  
--------------------------------------------------------------------------------

#Elasticnet
  
set.seed(123)
elasticnet<-train(song_popularity~., data = train, method = "glmnet", 
                  trControl = custom,
                  tuneGrid = expand.grid(alpha = seq(0.1, 1, length = 5), 
                                         lambda = seq(0.001, 1, length = 5)))

summary(elasticnet)  
elasticnet$results  
elasticnet$bestTune 

#alpha = 1, lambda = 0.00100, RMSE = 21.35

plot(varImp(elasticnet, scale = TRUE))    

--------------------------------------------------------------------------------

#Step6: Compare the models

compare<-list(linear = cv, ridge = ridge, lasso = lasso, elastic = elasticnet)
compare<-resamples(compare)
summary(compare)

#From comparison, we understood that Ridge is the best model with low RMSE or MAE
#However we shall predict all the models on test data and see. 

--------------------------------------------------------------------------------

#Step7: Predict the model 

install.packages("Metrics")
library(Metrics)

p1<-predict(elasticnet, test)

final<-data.frame(Actuals=test$song_popularity, Predicted=p1)
rmse<-RMSE(final$Actuals,final$Predicted)
rmse


#RMSE is 21.61 with Linear via 10-fold cross validation method,
#however we recommend to perform Decision Tree and Random forest before conlcusion. 