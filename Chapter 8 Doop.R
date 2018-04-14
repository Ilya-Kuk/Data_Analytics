set.seed(1)
#import package to construct classification and regression trees
install.packages("tree",repos='http://cran.us.r-project.org')
library(tree)
#get and restrucute carseats data
library(ISLR)

#Classification

attach(Carseats)
High <- ifelse(Sales<=8,"No","Yes")
Carseats <- data.frame(Carseats,High)
#make tree
tree.carseats <- tree(High~.-Sales,Carseats)
#look at tree
summary(tree.carseats)
plot(tree.carseats) #display structure
text(tree.carseats,pretty=0) #display node labels ##pretty=0 -> include category names for qualitative predictors
tree.carseats #output to each branch
#estimate test error rate
set.seed(2)
train=sample(1:nrow(Carseats), 200) #random index of 200 from all rows of data
Carseats_test <- Carseats[-train,] #test X
High_test <- High[-train] #test y
tree.carseats_train <- tree(High~.-Sales, Carseats, subset=train) #train tree
#look at trained tree
summary(tree.carseats_train)
plot(tree.carseats_train) #display structure
text(tree.carseats_train,pretty=0) #display node labels ##pretty=0 -> include category names for qualitative predictors
#test trained tree
tree.pred_test <- predict(tree.carseats_train,Carseats_test,type="class") #predicting on test
table(tree.pred_test,High_test) #confusion matrix
print(paste("The test error rate is",((27+30)/(nrow(Carseats_test)))))
#pruning tree
##should we prune tree? 
?cv.tree() ##performs K-fold cross-validation experiment to find the deviance or number of misclassifications as a function of cost-complexity parameter k.
##finds optimal tree complexity
##use FUN=prune.misclass to indicate we want classification error rate to guide cross-validation process, rather than the default FUN which is deviance
##note, 'dev' is the cross-validation error cause of this, NOT deviance
##reports number of terminal nodes of each tree considered as size, as well as corresponding error rate, and value of cost-complexity parameter - k
set.seed(3)
cv.carseats_train <- cv.tree(tree.carseats_train, FUN=prune.misclass)
names(cv.carseats_train)
cv.carseats_train #best is size 9, with 50 CV errors
#plot error as function of size and k
par(mfrow=c(1,2))
plot(cv.carseats_train$size, cv.carseats_train$dev, type="b") #type="b" makes it have lines
plot(cv.carseats_train$k, cv.carseats_train$dev, type="b")
dev.off()
#prune tree according to the cv
prune.carseats_train <- prune.misclass(tree.carseats_train, best=9)
plot(prune.carseats_train)
text(prune.carseats_train)
#test trained (pruned) model
tree.pred_test <- predict(prune.carseats_train, Carseats_test, type="class")
t <- table(tree.pred_test, High_test)
t
print(paste("The test error is", (t[1,1]+t[2,2])/nrow(Carseats_test))) #note, increasing best will lead to larger pruned tree with worse classification accuracy

# Regression

library(MASS)
set.seed(1)
#make trained tree
train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston_train <- tree(medv~., Boston, subset=train)
summary(tree.boston_train) #lstat measures percentage of individuals with lower socioeconomic status #only 3 variables used #in regression, deviance is sum of squared errors
#plot trained tree
plot(tree.boston_train)
text(tree.boston_train)
#see if pruning will be effective
cv.boston_train <- cv.tree(tree.boston_train)
plot(cv.boston_train$size, cv.boston_train$dev, type='b') #wants most complex tree
#pruning tree (anyway)
prune.boston_train <- prune.tree(tree.boston_train, best=5)
plot(prune.boston_train)
text(prune.boston_train)
#make predictions (using unpruned [better] tree)
tree.boston_pred <- predict(tree.boston_train, newdata=Boston[-train,])
boston_test <- Boston[-train,"medv"]
plot(tree.boston_pred,boston_test)
abline(0,1)
mean((tree.boston_pred-boston_test)^2)
print("Test MSE is 25.05. Root MSE is about 5.005, indicating this model leads to test predictions within around $5,005 of the true median home value for suburb.")

#Bagging and Random Forest

