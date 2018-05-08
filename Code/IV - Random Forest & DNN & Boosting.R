
rm(list=ls())
library(MASS)

# X and error are correlated (e.g. drawn from multivariate normal with same variance=1)
# Mean X=20, Mean error= 15, both have variance=1
x_and_e <- mvrnorm(1000, c(20, 15), matrix(c(1, 0.5, 0.5, 1), 2, 2))
endog_x <- x_and_e[, 1]
e <- x_and_e[, 2]


# Function That Specifies Relationship Between X and instrument
IV <- rnorm(1000)
#IV <- sample(1:10, 10000, replace=T)
x <- endog_x + IV
    # Alternate spects for IV (less correlated)
#x <- endog_x + .001*exp(IV)^3


# True definition of y (function of obsserved x)
y <- 1 + x + e

SimData<- as.data.frame(cbind(e, endog_x, IV, x, y))

# Check Correlations
cor(SimData$x, SimData$e)
cor(SimData$IV, SimData$e)
cor(SimData$x,SimData$IV)
cor(SimData$e,SimData$y)

# True effect of x on y
lm(y ~ x+e , data=SimData)
# observed effect of x on y
lm(y ~ x , data=SimData)
# IV estimated effect of X on y
xHat_LM <- lm(x ~ IV, data=SimData)$fitted.values
SimData<- as.data.frame(cbind(xHat_LM, SimData))
# IV Model
lm(y ~ xHat_LM, data=SimData)
#summary(lm(y ~ xHat_LM, data=SimData))



################################################## Using Caret
library(caret)
# Split The Data Into Train & Test Sets
index <- createDataPartition(SimData$y, p=0.75, list=FALSE)
SimData_Train <- SimData[ index,]
SimData_Test <- SimData[-index,]

#SimData_Train = subset(SimData_Train, select = c(IV,x))
#SimData_Test = subset(SimData_Test, select = c(IV,x))



#######  Gradient Boosting
library(gbm)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbm_Fit1 <- train(x ~ IV, data = SimData_Train, 
                 method = "gbm", 
                 trControl = fitControl,
                 #tuneGrid = gbmGrid,
                 verbose = FALSE)

gbm_Fit1
plot(gbm_Fit1) 


xHat_GBM = predict(gbm_Fit1, newdata = SimData_Test)

SimData_Test<- as.data.frame(cbind(xHat_GBM, SimData_Test))
# IV Model
lm(y ~ xHat_GBM, data=SimData_Test)
#summary(lm(y ~ xHat_LM, data=SimData))


########################### DNN ###########################
numFolds <- trainControl(method = 'cv', 
                         number = 10, 
                         #classProbs = TRUE,  - Only relevant for categorical data
                         #verboseIter = TRUE, 
                         #summaryFunction = twoClassSummary, 
                         preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))


DNN_fit <- train(x ~ IV, data = SimData_Train, 
             method = 'nnet', 
             trControl = numFolds,
             tuneGrid=expand.grid(size=c(10), decay=c(0.1))
             )

DNN_fit

xHat_DNN = predict(DNN_fit, newdata = SimData_Test)

SimData_Test<- as.data.frame(cbind(xHat_DNN, SimData_Test))
# IV Model
lm(y ~ xHat_DNN, data=SimData_Test)


########################### SVM ###########################
#http://dataaspirant.com/2017/01/19/support-vector-machine-classifier-implementation-r-caret-package/
#library(e1071)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)

svm_Linear <- train(x ~ IV, data = SimData_Train, 
                    method = "svmLinear",
                    trControl=fitControl,
                    tuneLength = 10)

svm_Linear

xHat_SVM <- predict(svm_Linear, newdata = SimData_Test)
lm(y ~ xHat_SVM, data=SimData_Test)


svm_radial <- train(x ~ IV, data = SimData_Train, 
                    method = "svmRadial",
                    trControl=fitControl,
                    tuneLength = 10)

svm_radial

xHat_SVM_rad <- predict(svm_radial, newdata = SimData_Test)
lm(y ~ xHat_SVM_rad, data=SimData_Test)








#######  Random Forrest Model
library(randomForest)
#https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/


modelLookup(model='rf')


fitControl <- trainControl(
  ## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## Repeated 10 times
  repeats = 10)

RF_Fit1 <- train(x ~ IV, data = SimData_Train, 
                 method = "ranger",
                 trControl = fitControl
                 #,verbose = FALSE
)

xHat_RF <- predict(RF_Fit1, newdata = SimData_Test)
lm(y ~ xHat_RF, data=SimData_Test)

warnings()
