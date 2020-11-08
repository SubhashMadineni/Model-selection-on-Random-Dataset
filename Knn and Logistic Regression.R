#########################################################
## data Mining I HW4
## Subhashchandra Babu Madineni   UBIT = 50373860
## Created on 23 th october
## Edited: 
#########################################################
rm(list = ls())

#setwd('C:/University At Buffalo Fall 2020 Classes/EAS 506-CDA 541Stastical Data Mining/R for STA/homework_4')
#library(ISLR)
library(MMST)
library(MASS)
library(klaR)   #install.packages("MMST")
library(mclust)   #install.packages("mclust")
library(rda)
library(tidyverse)
library(ggplot2)
require(class)
library(class)
library(pls)
library(ISLR)
library(caret)

dataset = Weekly

pairs(dataset)

#########################################
# Create a training and test set
#########################################
set.seed(63)
train <- sample(1:nrow(dataset), .75*nrow(dataset))
X_train<- dataset[train,]
X_test <- dataset[-train,]

y_true_train <- as.numeric(X_train$Direction)-1
y_true_test <- as.numeric(X_test$Direction)-1


#########################################
# Applying Logistic Regression
#########################################
log_reg <- glm(Direction ~., data = X_train, family = "binomial")
summary(log_reg)

# Predicting the train and test Data
glm.probs.train <- predict(log_reg, newdata = X_train, type = "response")
y_hat_train <- round(glm.probs.train)
glm.probs.test <- predict(log_reg, newdata = X_test, type = "response")
y_hat_test <- round(glm.probs.test)

#cbind(y_true_train, y_hat_train)

#  Calculate the error rates
train_err <- sum(abs(y_hat_train- y_true_train))/length(y_true_train)
test_err <- sum(abs(y_hat_test- y_true_test))/length(y_true_test)

train_err    # 0
test_err    #0.00366


#########################################
#  Confusion Matrix
########################################
conf <- confusionMatrix(as.factor(y_hat_test), as.factor(y_true_test))
names(conf)
conf$table

missclass_rate_log_reg_full =  (conf$table[1,2]+conf$table[2,1])/length(y_hat_test)*100
missclass_rate_log_reg_full   # 0.36%




#####################################################################################################################
#-------------------D fitting Lag2 Varriable with 1990-2008 Data-----------------------------------------
#####################################################################################################################




#########################################
# Create a training and test set
#########################################
train_1990_2008 = dataset[1:985,]
test_2009_2010 = dataset[986:1089,]
y_true_train_1990_2008 <- as.numeric(train_1990_2008$Direction)-1
y_true_test_2009_2010 <- as.numeric(test_2009_2010$Direction)-1


#########################################
# Applying Logistic Regression on new data
#########################################
set.seed(435)
log_reg_90_08 <- glm(Direction ~Lag2, data = train_1990_2008, family = "binomial")
summary(log_reg_90_08)



#########################################
# Predicting the train and test Data
#########################################
glm.probs.train_1990_2008 <- predict(log_reg_90_08, newdata = train_1990_2008, type = "response")
y_hat_train_1990_2008 <- round(glm.probs.train_1990_2008)
glm.probs.test_2009_2010<- predict(log_reg_90_08, newdata = test_2009_2010, type = "response")
y_hat_test_2009_2010 <- round(glm.probs.test_2009_2010)

#########################################
#  Calculate the error rates
########################################
train_1990_2008_err <- sum(abs(y_hat_train_1990_2008- y_true_train_1990_2008))/length(y_true_train_1990_2008)
test_2009_2010_err <- sum(abs(y_hat_test_2009_2010- y_true_test_2009_2010))/length(y_true_test_2009_2010)

train_1990_2008_err    #0.4446701
test_2009_2010_err     #0.375


#########################################
#  Confusion Matrix
########################################
conf_2<- confusionMatrix(as.factor(y_hat_test_2009_2010), as.factor(y_true_test_2009_2010))
conf_2$table

missclass_rate_log_reg =  (conf_2$table[1,2]+conf_2$table[2,1])/length(y_hat_test_2009_2010)*100
missclass_rate_log_reg   # 37.5%



############################################################################################################
#-------------------D fitting LDA to Lag2 Varriable with 1990-2008 Data-----------------------------------------
############################################################################################################

#########################################################
#  Applying LDA
########################################################

lda.fit <- lda(Direction~Lag2, data = train_1990_2008)
summary(lda.fit)
lda.pred.train <- predict(lda.fit, newdata = train_1990_2008)
y_hat_train_LDA <- as.numeric(lda.pred.train$class)-1
lda.pred.test <- predict(lda.fit, newdata = test_2009_2010)
y_hat_test_LDA <- as.numeric(lda.pred.test$class)-1

# Compute the error
lda_train_error <- sum(abs(y_true_train_1990_2008 - y_hat_train_LDA))/length(y_true_train_1990_2008)
lda_test_error <- sum(abs(y_true_test_2009_2010 - y_hat_test_LDA))/length(y_true_test_2009_2010)
lda_train_error    #0.4456853
lda_test_error     #0.375



#########################################
#  Confusion Matrix
########################################
conf_LDA<- confusionMatrix(as.factor(y_hat_test_LDA), as.factor(y_true_test_2009_2010))
names(conf_LDA)
conf_LDA$table

missclass_rate_lda =  (conf_LDA$table[1,2]+conf_LDA$table[2,1])/length(y_hat_test_LDA)*100
missclass_rate_lda   # 37.5%

############################################################################################################
#-------------------D fitting KNN to Lag2 Varriable with 1990-2008 Data-----------------------------------------
############################################################################################################

#########################################
# Create a training and test set
#########################################
train_knn = dataset[1:985,3]
test_knn = dataset[986:1089,3]
y_true_train_1990_2008 <- as.numeric(train_1990_2008$Direction)-1
y_true_test_2009_2010 <- as.numeric(test_2009_2010$Direction)-1

train_1990_2008$Direction <- as.numeric(train_1990_2008$Direction)-1

test_2009_2010$Direction <- as.numeric(test_2009_2010$Direction)-1

################################################
# Building the model for knn
###############################################
y_pred_knn = (knn(train = train_1990_2008,test = test_2009_2010,cl = y_true_train_1990_2008,k=1))


conf_knn = table(y_true_test_2009_2010,y_pred_knn)
conf_knn
missclass_rate_knn =  (conf_knn[1,2]+conf_knn[2,1])/length(y_pred_knn)*100
missclass_rate_knn   # 18.26923%


