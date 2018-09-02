#trainingData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))


#testData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

##ENVIRONMENT SETUP
dataDir <- "data"
trainingFile <-"./data/trainingFile.csv"
trainingFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testFile <- "./data/testFile.csv"
testFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

setwd("~/CURSODATA/coursera/08_PracticalMachineLearning")
if (!dir.exists(dataDir)){  dir.create(dataDir)}
if (!file.exists(trainingFile)){download.file(trainingFileURL ,destfile=trainingFile,method="auto")}
if (!file.exists(testFile)){download.file(testFileURL ,destfile=testFile,method="auto")}

testData <- read.csv(testFile)
trainingData <- read.csv(trainingFile)

#Se trata de un proyecto en los que se levantan pesas con tres sensores, uno en la citura, en el brazo y en la muñeca 
#y un cuarto sensor en la pesa. cada sensor toma distintas mediciones (acelerómetro, giroscopio, etc..). En el dataset de
#training se ha clasificado según estos sensores la clase de ejercicio que se está haciendo, siendo únicamente el ejercicio A
# el correcto, el resto se consideran movimientos erróneos.Se pide crear el modelo en base a este training set y hacer las valoraciones
# correspondientes sobre el testSet. Hay que hacer primero una reducción de dimensiones para saber qué variables son necesarias en 
# nuestro modelo para luego elegir un método predictivo.

list.of.packages <- c("knitr","caret","rpart","rpart.plot","rattle","randomForest","corrplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
set.seed(301)

##DATA CLEANING
# remove variables with Nearly Zero Variance
NZV <- nearZeroVar(trainingData)
TrainSet <- trainingData[, -NZV]
TestSet  <- testData[, -NZV]
dim(TestSet)

# remove variables that are mostly NA
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TestSet)

# remove identification only variables (columns 1 to 6)
TrainSet <- TrainSet[, -(1:6)]
TestSet  <- TestSet[, -(1:6)]
dim(TrainSet)

corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

#If two variables are highly correlated their colors are either dark blue (for a positive correlation) 
#or dark red (for a negative corraltions). To further reduce the number of variables, a Principal Components Analysis (PCA) 
#could be performed as the next step. However, since there are only very few strong correlations 
#among the input variables, the PCA will not be performed. 
#Instead, a few different prediction models will be built next.

#Partitioning the Dataset
#Following the recommendation in the course Practical Machine Learning, we will split our data into a training data set (70% of the total cases) and a testing data set (40% of the total cases; the latter should not be confused with the data in the pml-testing.csv file). This will allow us to estimate the out of sample error of our predictor.
set.seed(12345)

# create a partition using caret with the training dataset on 70,30 ratio
inTrain  <- createDataPartition(TrainSet$classe, p=0.7, list=FALSE)

training <- TrainSet[inTrain, ]

testing  <- TrainSet[-inTrain, ]
dim(TrainSet)

dim(training); dim(testing);

#Three popular methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below. 
#A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

#Decision Tree
set.seed(12345)
modFitDT <- rpart(classe ~ ., data = training, method="class", control = rpart.control(method = "cv", number = 10))
fancyRpartPlot(modFitDT)

#Predicting
set.seed(12345)
prediction <- predict(modFitDT, testing, type = "class")
confMatDecTree <-confusionMatrix(prediction, testing$classe)
confMatDecTree

# plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))

#RANDOM FOREST
#Using random forest, the out of sample error should be small. The error will be estimated using the 40% testing sample. We should expect an error estimate of < 3%.

#buiding the model
set.seed(12345)
modFitRF <- randomForest(classe ~ ., data = training, method = "rf", importance = T, trControl = trainControl(method = "cv", classProbs=TRUE,savePredictions=TRUE,allowParallel=TRUE, number = 10))
plot(modFitRF)

#predicting
prediction <- predict(modFitRF, testing, type = "class")
confusionMatrix(prediction, testing$classe)


# model fit
set.seed(301)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=training, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel

# prediction on Test dataset
predictRandForest <- predict(modFitRandForest, newdata=testing)
confMatRandForest <- confusionMatrix(predictRandForest, testing$classe)
confMatRandForest

#plot matrix results
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))


#GBM GENERALIZED BOOSTED MODEL
modFitBoost <- train(classe ~ ., method = "gbm", data = training,
                     verbose = F,
                     trControl = trainControl(method = "cv", number = 10))

modFitBoost
plot(modFitBoost)
#Predicting
prediction <- predict(modFitBoost, testing)
confusionMatrix(prediction, testing$classe)

# model fit
set.seed(301)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=training, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)

modFitGBM$finalModel

# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=testing)
confMatGBM <- confusionMatrix(predictGBM, testing$classe)
confMatGBM

# plot matrix results
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))

#5. Applying the selected Model to the Test Data
#The accuracy of the 3 regression modeling methods above are:
  
#  Random Forest : 0.9968 Decision Tree : 0.8291 GBM : 0.9884 In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.

predictTEST <- predict(modFitRandForest, newdata=TestSet)
predictTEST


features <- names(trainingData[,colSums(is.na(trainingData)) == 0])
features <- names(trainingData[, -nearZeroVar(trainingData)])
features2 <- names(trainingData[,!sapply(trainingData, function(x) mean(is.na(x))) > 0.95])

tra <- trainingData[,features]

tra <- tra[,features2]

# Only use features used in testing cases.
dt_training <- trainingData[,c(features,"classe")]



goodCol <- colSums(is.na(trainingData)) < 19600
myTraining <- trainingData[ , goodCol][ , ]
myTraining <- myTraining[ , -(1:7)]
