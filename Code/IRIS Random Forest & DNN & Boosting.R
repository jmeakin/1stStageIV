
########################### Random Forrest Model ###########################
rm(list=ls())
library(datasets)
data(iris)


####### Divide the Sample
  # Question For Chena: presumably here, we are sampling 30% and and 70% with 
ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

#######  Random Forrest Model
library(randomForest)
iris_rf <- randomForest(Species~.,data=trainData,ntree=1000,proximity=TRUE)
table(predict(iris_rf),trainData$Species)
print(mean(iris_rf$err.rate[,1]))

irisPred<-predict(iris_rf,newdata=testData)
table(irisPred, testData$Species)
irisPred$err.rate

iris_rf[[7]][5]
iris_rf$err.rate[100,1]


########################## Simulation - Forest
rm(list=ls())
library(datasets)
data(iris)
iterations = 200
test_set = .80
vector = c()
for (i in 1:iterations) {
 ind <- sample(2,nrow(iris),replace=TRUE,prob=c(test_set,1-test_set))
 trainData <- iris[ind==1,]
 testData <- iris[ind==2,]
 iris_rf <- randomForest(Species~.,data=trainData,ntree=100,proximity=TRUE)
 print(mean(iris_rf$err.rate[,1]))
 mean <- mean(iris_rf$err.rate[,1])
 vector <- append(vector,mean)
}
mean(vector)






########################### Gradient Boosting ###########################
# http://allstate-university-hackathons.github.io/PredictionChallenge2016/GBM
rm(list=ls())
#install.packages('gbm')
library(datasets)
library(gbm)
data(iris)


####### Divide the Sample
ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]


iris_gbm <-    gbm(Species~. ,
               data = trainData,
               distribution = "gaussian",
               n.trees = 1000,
               shrinkage = 0.01, 
               interaction.depth = 4)
iris_gbm
summary(iris_gbm)

gbmtestPredictions <- predict(object = iris_gbm,
                              newdata = testData,
                              n.trees = 1000,
                              type = "response")


head(data.frame("Actual" = testData$Species, 
                "PredictedProbability" = gbmtestPredictions))















########################### DNN ###########################
rm(list=ls())
library(datasets)
library(keras)
#install_tensorflow()
#https://www.datacamp.com/community/tutorials/keras-r-deep-learning#model

####### Load Data
data(iris)
#######  Make Strings Numeric
iris[,5] <- as.numeric(iris[,5]) -1
iris <- as.matrix(iris)
dimnames(iris) <- NULL
#iris <- normalize(iris[,1:4])
#######  Determine sample size
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(.70, .30))
# This just creates a data data frame with 70% of the rows as 1, the rest 2
#######  Split the `iris` data
iris.training_X <- iris[ind==1, 1:4]
iris.test_X <- iris[ind==2, 1:4]
# this says, take the first 4 columns of the data and the 1/2s identify which data is in each set
#######   Split the class attribute (same as above)
iris.trainingtarget_y <- iris[ind==1, 5]
iris.testtarget_y <- iris[ind==2, 5]
####### Make Categorical( 0/0/1, 0/1/0, 1/0/0)
#like onehot in Python
iris.trainLabels_y <- to_categorical(iris.trainingtarget_y)
iris.testLabels_y <- to_categorical(iris.testtarget_y)


#######  Initialize a sequential model
model <- keras_model_sequential()

#######  Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')


#######  Print a summary of a model
summary(model)
#######  Get model configuration
get_config(model)
#######   Get layer configuration
get_layer(model, index = 1)
####### List the model's layers
model$layers
####### List the input tensors
model$inputs
####### List the output tensors
model$outputs


#######   Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

#######    Fit the model 
model %>% fit(
  iris.training_X, 
  iris.trainLabels_y, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)



#######     Store the fitting history in `history` 
history <- model %>% fit(
  iris.training_X, 
  iris.trainLabels_y, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)


# Predict the classes for the test data
classes <- model %>% predict_classes(iris.test_X, batch_size = 128)
table(iris.testtarget_y, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(iris.test_X, iris.testLabels_y, batch_size = 128)

# Print the score
print(score)
print(score$acc)


########################## Simulation - DNN
rm(list=ls())
library(datasets)
library(keras)
data(iris)

iterations = 10
test_set = .80
# model parameters
epoc = 10

vector = c()
for (i in 1:iterations) {
  data(iris)
  # Clean Data
  iris[,5] <- as.numeric(iris[,5]) -1
  iris <- as.matrix(iris)
  dimnames(iris) <- NULL
  ind <- sample(2, nrow(iris), replace=TRUE, prob=c(test_set, 1-test_set))
  iris.training_X <- iris[ind==1, 1:4]
  iris.test_X <- iris[ind==2, 1:4]
  iris.trainingtarget_y <- iris[ind==1, 5]
  iris.testtarget_y <- iris[ind==2, 5]
  iris.trainLabels_y <- to_categorical(iris.trainingtarget_y)
  iris.testLabels_y <- to_categorical(iris.testtarget_y)
  
  # Define & Compile Model
  model <- keras_model_sequential()
  model %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
    layer_dense(units = 3, activation = 'softmax')
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  #Fit the model 
  model %>% fit(
    iris.training_X, 
    iris.trainLabels_y, 
    epochs = epoc, 
    batch_size = 5, 
    validation_split = 0.2
  )
  # Score Model
  score <- model %>% evaluate(iris.test_X, iris.testLabels_y, batch_size = 128)
  acuracy <- 1-score$acc
  vector <- append(vector,acuracy)
  
}
mean(vector)





########################### SVM ###########################
# https://eight2late.wordpress.com/2017/02/07/a-gentle-introduction-to-support-vector-machines-using-r/
# https://rischanlab.github.io/SVM.html

rm(list=ls())
#install.packages('e1071')
library(e1071)
library(datasets)
data(iris)
attach(iris)


svm_model <- svm(Species ~ ., data=iris)
summary(svm_model)

x <- subset(iris, select=-Species)
y <- Species
svm_model1 <- svm(x = x, y = y)
summary(svm_model1)


pred <- predict(svm_model,x)
table(pred,y)


