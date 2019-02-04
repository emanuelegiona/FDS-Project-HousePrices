# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Main script merging the individual work done.

# Libraries
library(plyr)
library(dplyr)

require(glmnet)
require(Matrix)
require(xgboost)
require(kernlab)
library(caret)

library(e1071)
library(psych)

source("featureEngineering.R")
source("modelTesting.R")

# Datasets
train   <- read.csv("data/train.csv", stringsAsFactor=FALSE)
test    <- read.csv("data/test.csv", stringsAsFactor=FALSE)

# Preliminary data handling
testIDs  <- test$Id     # Save the Ids for submission
train$Id <- NULL        # and remove them in the dataset
test$Id  <- NULL
test$SalePrice <- NA    # Test hasn't any SalePrice, then set it as NA

# Complete dataset (train + test sets)
fullData <- rbind(train, test)

# Performs data tidying and feature engineering
bootstrap <- function(data, totBathRms=T, carsXarea=T, recentGarage=F, totalSF=T){
    data <- removeOutliers(data)
    data <- handleSkewness(data)
    
    futureDeletionNames <- getFactorFields(data)
    data <- handleNA(data)
    data <- appendDummyVariables(data)
    data[futureDeletionNames] <- NULL
    
    data <- addNewFeatures(data, totBathRms, carsXarea, recentGarage, totalSF)
    
    data <- replaceRemainingNAwtMean(data)
    data <- removeMulticollinearFeatures(data)
}

# Performs feature selection
getFinalFeatures <- function(data){
    data <- bootstrap(data)
    data <- importanceSelection(data, getLassoModel)
    data
}

# Writes predictions to a Kaggle-compliant CSV file
#
# Parameters:
# ===========
#   -`ids`:         IDs of test set entries
#   -`pred`:        predictions of test set SalePrice values
#   -`outputPath`:  string containing a path to write the file to (comprehensive file extension)
savePredictionsOnFile <- function(ids, pred, outputPath){
    predictionDF <- data.frame(Id = ids, SalePrice = pred)
    write.csv(predictionDF, file = outputPath, row.names = FALSE)
}

# Trains models, performs predictions and writes a Kaggle-compliant CSV file
makeFinalPredictions <- function(){
    print("Cleaning dataset...")
    fullData <- getFinalFeatures(fullData)
    
    print("Training the optimal ensemble model...")
    # optimal ensemble model
    models <- c(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    weights <- c(0.5, 0.5, 3.5, 5)
    
    ensemble <- createEnsembleModel(models, weights, fullData)
    ensemblePreds <- ensemblePredict(ensemble, fullData)
    
    print("Training XGB model...")
    # XGB
    xgb <- getXGBModel(fullData)
    xgbPreds <- predictSalePrices(xgb, fullData)
    
    print("Training SVM model...")
    # SVM
    svm <- getSVM(fullData)
    svmPreds <- predictSalePrices(svm, fullData)
    
    # Stacked regressors
    recipe <- list(getLassoModel, getRidgeModel, getXGBModel, getSVM)
    
    print("Training stacked regressor, variant A...")
    stackedA <- getStackedRegressor(fullData, recipe, getXGBModel, variant = "A")
    stackPredsA <- stackedA$predictions
    
    print("Training stacked regressor, variant B...")
    stackedB <- getStackedRegressor(fullData, recipe, getXGBModel, variant = "B")
    stackPredsB <- stackedB$predictions
    
    print("Averaging predictions...")
    # final predictions
    preds <- (2*ensemblePreds + xgbPreds + svmPreds + stackPredsA + stackPredsB)/6
    
    print("Writing CSV...")
    # write CSV
    savePredictionsOnFile(testIDs, preds, "out/final_predictions.csv")
    
    print("Done.")
}
