# Authors:  Angelo Di Mambro, Emanuele Giona, Luigi Berducci
# Date:     January 2019
# Purpose:  Create a main script which merge the individual work done.
#           Remember that the script is shared among the students on Github.
#           Then before commit the update, invoke "git stash", "git pull" and "git stash pop".

# Libraries
library(ggplot2)
library(plyr)
library(dplyr)
library(Boruta)
library(corrplot)
library(scales)
library(Rmisc)
library(ggrepel)
library(psych)
library(xgboost)
library(caret)

# Dataset
train   <- read.csv("data/train.csv", stringsAsFactor=FALSE)
test    <- read.csv("data/test.csv", stringsAsFactor=FALSE)

# Preliminary data handling
testIDs  <- test$Id     # Save the Ids for submission
train$Id <- NULL        # and remove them in the dataset
test$Id  <- NULL
test$SalePrice <- NA    # Test hasn't any SalePrice, then set it as NA

fullData <- rbind(train, test)

# Features' blocks (NEVER USED, TODO: remove it if unnecessary)
location <- c("MSSubClass", "MSZoning", "Street", "Alley", "Neighborhood", "Condition1", "Condition2")
lot <- c("LotFrontage", "LotArea", "LotShape", "LandContour", "LotConfig", "LandSlope")
misc <- c("Utilities", "BldgType", "HouseStyle", "Heating", "HeatingQC", "CentralAir", "Electrical", "Fireplaces", "FireplaceQu", "MiscFeature", "MiscVal")
outside = c("PavedDrive", "Fence", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch") 
kitchen = c("KitchenAbvGr", "KitchenQual")
bathrooms = c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")
general = c("OverallQual", "OverallCond", "Functional")
rooms = c("BedroomAbvGr", "TotRmsAbvGrd")
#POOL AND GARAGE ARE USED IN CHECK CONSISTENCY METHODS
pool <- c("PoolArea", "PoolQC")
garage = c("GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond")

luigi  <- c(general, bathrooms, kitchen, rooms, garage, outside, 'SalePrice')
significative_luigi <- c("OverallQual", "FullBath", "HalfBath", "KitchenQual", "TotRmsAbvGrd", "GarageFinish", "GarageCars", "GarageArea", "GarageCarsTimesArea", "TotBathRms", "RecentGarage1", "GarageRecentType", "GarageTypeOT")
angelo <- c("YearBuilt","YearRemodAdd","MoSold","YrSold","SaleType","SaleCondition","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","LowQualFinSF","GrLivArea","SalePrice")

# List of Ordinal variables conversion
AccessType  <- c("None" = 0, "Grvl" = 1, "Pave" = 2)
LotShape    <- c("IR3" = 0, "IR2" = 1, "IR1" = 2, "Reg" = 3)
LandSlope   <- c("Sev" = 0, "Mod" = 1, "Gtl" = 2)
Utilities   <- c("None" = 0, "ELO" = 1, "NoSeWa" = 2, "NoSewr" = 3, "AllPub" = 4)
CentralAir  <- c("N" = 0, "Y" = 1)
Qualities   <- c('None'=0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
Exposure    <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
FinType     <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
Masonry     <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
PavedDrive  <- c('Y'=1, 'N'=0, 'P'=0)
Functional  <- c('Typ'=7, 'Min1'=6, 'Min2'=5, 'Mod'=4, 'Maj1'=3, 'Maj2'=2, 'Sev'=1, 'Sal'=0)
GarageFinish <- c('Fin'=3, 'RFn'=2, 'Unf'=1, 'Miss'=0)

# Control flag for skewness conversion
skewCorrection <- FALSE

# Helper Functions
#returns the most important features, estimated via the Boruta technique; can be set up to only work on selected features (default: all) and to meet a certain importance threshold (default: 0)
applyBoruta <- function(data, features = NULL, threshold = 0){
  set.seed(12345)
  if(!is.null(features)){
    data <- data %>%
      select(., SalePrice, features) %>%
      filter(., !is.na(.$SalePrice))
  }

  borutaImp <- Boruta(data$SalePrice~., data = data, doTrace = 0)
  goodFeatures <- getSelectedAttributes(borutaImp)
  stats <- attStats(borutaImp) %>%
    filter(., .$decision == "Confirmed")

  stats$feature <- goodFeatures
  stats <- stats %>%
    filter(.$meanImp >= threshold) %>%
    select(., feature, meanImp) %>%
    arrange(.$meanImp)
}

#returns the mode of a collection of values, ignoring NAs
getMode <- function(values){
  uniques <- unique(values)
  uniques <- uniques[!is.na(uniques)]
  maxFreqID <- which.max(tabulate(match(values,uniques)))
  uniques[maxFreqID]
}

#returns values as factors, filling NAs with the mode or with the specified argument
encodeAsFactor <- function(values, replaceNA = "NA"){
  if(!is.factor(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.factor(values)
  }
  values
}

#returns values as integers, enconding them via the given dictionary, filling NAs with the mode or with the specified argument
encodeAsOrdinal <- function(values, dictionary, replaceNA = "NA"){
  if(!is.integer(values))
  {
    values[is.na(values)] <- ifelse(replaceNA == "NA", getMode(values), replaceNA)
    values <- as.integer(dictionary[values])
  }
  values
}

#returns average frontage given a certain neighborhood
avgFrontage <- function(neighborhood){
  as.integer(mean(fullData$LotFrontage[fullData$Neighborhood == neighborhood], na.rm = T))
}

#substitues all NAs with the average frontage of the same neighborhood
getValidFrontages <- function(data){
  if(0 < length(which(is.na(data$LotFrontage))))
  {
    for(i in 1:nrow(data))
      if(is.na(data$LotFrontage[i]))
        data$LotFrontage[i] <- avgFrontage(data$Neighborhood[i])
  }
  data
}

#fixes incoherent MiscFeature and MiscVal values (MiscFeature = NA <-> MiscVal = 0)
getValidMiscFeaturesAndVal <- function(data){
  if(0 < length(which((is.na(data$MiscFeature) & data$MiscVal > 0) | (!is.na(data$MiscFeature) & data$MiscVal == 0))))
  {
    for(i in 1:nrow(data)){
      if(is.na(data$MiscFeature[i]) & data$MiscVal[i] > 0)
        data$MiscVal[i] <- 0
      else if(!is.na(data$MiscFeature[i]) & data$MiscVal[i] == 0)
        data$MiscFeature[i] <- "None"
    }
  }
  data
}

# Main functions
iterateCrossValidationNTimes <- function(data, nTimes){
    finalRes <- data.frame()
    for(i in 1:nTimes){
        currentRes <- crossValidation(data)
        finalRes <- rbind(finalRes, currentRes)
    }
    finalRes
}

crossValidation <- function(data){
    # Work only on train data
    data <- getOnlyRelevantFeatures(data)
    allTrain <- getTrainData(data) 

    # Split in train/test data
    trainSamples <- allTrain$SalePrice %>% createDataPartition(p=0.8, list=FALSE)
    trainData <- allTrain[trainSamples, ]
    testData  <- allTrain[-trainSamples, ]

    # Save the groundtruth to future comparison
    groundTruth <- testData$SalePrice
    testData$SalePrice <- NA

    # Build the model and predict prices
    model <- getSimpleLinearModel(trainData)
    pred  <- predictSalePrices(model, testData)
    res   <- data.frame( R2 = R2(pred, groundTruth),
                         RMSE = RMSE(pred, groundTruth),
                         MAE = MAE(pred, groundTruth))
    res
}

savePredictionsOnFile <- function(ids, pred, outputPath){
    predictionDF <- data.frame(Id = ids, SalePrice = pred)
    write.csv(predictionDF, file = outputPath, row.names = FALSE)
}

predictSalePrices <- function(model, data){
    test <- getTestData(data)
    test$SalePrice <- NULL
    predictions <- predict(model, test)
    if (skewCorrection==TRUE)
        predictions <- exp(predictions)
    predictions
}

getSimpleLinearModel <- function(data){
    train <- getTrainData(data)
    model <- lm(SalePrice ~ ., data=train)
    model
}

getOnlyRelevantFeatures <- function(data) {
    numerical <- removeFactors(data)
    # Maintain the list of features as clear as possible
    notRelevantEma <- c("MSSubClass", "Utilities", "Street", "Condition2")
    notRelevantAng <- c("X1stFlrSF","X2ndFlrSF","LowQualFinSF", "GrLivArea", "TotalBsmtSF", "YearBuilt", "YearRemodAdd")
    notRelevantLui <- c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "GarageYrBlt", "GarageCars", "GarageArea")
    notRelevant <- c(notRelevantAng, notRelevantEma, notRelevantLui)
    toRemove <- names(numerical) %in% notRelevant
    relevant <- numerical[!toRemove]
    relevant
}

getTrainData <- function(data){
    train <- data[!is.na(data$SalePrice), ]
    train
}

getTestData <- function(data){
    test <- data[is.na(data$SalePrice), ]
    test
}

featureEngineering <- function(data){
    #Emanuele
    data <- handleLocations(data) 
    data <- handleLot(data) 
    data <- handleMisc(data) 
    #Angelo
    data <- handleSaleBsmtAndYears(data)
    #Luigi
    data <- handleGarage(data)
    data <- handleOutside(data)
    data <- handleRooms(data)
    data <- addFeatureBathrooms(data)
    data <- addFeatureRecentGarage(data)
    data <- addFeatureCarsXArea(data)
    data <- addFeatureRecentType(data)

    data <- correctSkewness(data)
    # data <- appendDummyVariables(data)
    data
}

correctSkewness <- function(data){
    #Correct skewness on prices
    skewCorrection <- TRUE
    data$SalePrice <- log(data$SalePrice)
    #Correct skewness on other fields
    # TODO
    data
}

removeFactors <- function(data){
    factNames <- getFactorFields(data)
    data <- data[!(names(data) %in% factNames)]
    data
}

appendDummyVariables <- function(data){
    prices <- data$SalePrice
    data$SalePrice <- NULL
    fact <- getFactorData(data)
    dummies <- as.data.frame(model.matrix(~.-1, fact))
    data <- cbind(data, dummies)
    data$SalePrice <- prices
    data
}

handleRooms <- function(data){
    data$KitchenQual <- encodeAsOrdinal(data$KitchenQual, Qualities)
    data$Functional <- encodeAsOrdinal(data$Functional, Functional, "Typ")
    data$KitchenAbvGr[is.na(data$KitchenAbvGr)] <- 0
    data$FullBath[is.na(data$FullBath)] <- 0
    data$HalfBath[is.na(data$HalfBath)] <- 0
    data$BsmtFullBath[is.na(data$BsmtFullBath)] <- 0
    data$BsmtHalfBath[is.na(data$BsmtHalfBath)] <- 0
    data
}

handleGarage <- function(data){
    data <- checkConsistencyGarage(data)
    data$GarageFinish <- encodeAsOrdinal(data$GarageFinish, GarageFinish, "Miss" )
    data$GarageQual <- encodeAsOrdinal(data$GarageQual, Qualities, "None" )
    data$GarageCond <- encodeAsOrdinal(data$GarageCond, Qualities, "None" )
    data$GarageCars[is.na(data$GarageCars)] <- 0
    data$GarageYrBlt[is.na(data$GarageYrBlt)] <- 0
    data$GarageArea[is.na(data$GarageArea)] <- 0
    data$GarageType[!(data$GarageType == 'BuiltIn' | data$GarageType=='Attchd')] <- 'OT'
    data$GarageType[data$GarageType == 'BuiltIn' | data$GarageType=='Attchd'] <- 'BA'
    data$GarageType <- encodeAsFactor(data$GarageType, "OT")
    data
}

handleOutside <- function(data){
    data <- checkConsistencyPool(data)
    data$PavedDrive <- encodeAsOrdinal(data$PavedDrive, PavedDrive, "N")
    data$Fence <- encodeAsFactor(data$Fence, "None")
    data$PoolQC <- encodeAsOrdinal(data$PoolQC, Qualities, "None")
    data$PoolArea[is.na(data$PoolArea)] <- 0
    data
}


checkConsistencyGarage <- function(data){
    data$GarageArea[data$GarageArea==0] <- NA
    data$GarageCars[data$GarageCars==0] <- NA
    for(i in 1:nrow(data))
        if(currentRowIsGarageInconsistent(data, i)) {
            data$GarageType[i] <- NA
            data$GarageQual[i] <- NA
            data$GarageCond[i] <- NA
            data$GarageCars[i] <- NA
            data$GarageArea[i] <- NA
            data$GarageYrBlt[i] <- NA
            data$GarageFinish[i] <- NA
        }
    data
}

checkConsistencyPool <- function(data){
    data$PoolArea[data$PoolArea==0] <- NA
    for(i in 1:nrow(data))
        if(currentRowIsPoolInconsistent(data, i)) {
            data$PoolQC[i] <- NA
            data$PoolArea[i] <- NA
        }
    data
}

currentRowIsGarageInconsistent <- function(data, i){
    currentRowIsInconsistentGivenSetOfFeatures(data, garage, i)
}

currentRowIsPoolInconsistent <- function(data, i){
    currentRowIsInconsistentGivenSetOfFeatures(data, pool, i)
}

currentRowIsInconsistentGivenSetOfFeatures <- function(data, features, i){
    currentData <- data[i, features]
    isNA <- sapply(currentData, is.na)
    TRUE %in% isNA & FALSE %in% isNA
}

handleSaleBsmtAndYears <- function(data){
    data$MoSold         <- encodeAsFactor(data$MoSold)
    data$SaleType       <- encodeAsFactor(data$SaleType)
    data$SaleCondition  <- encodeAsFactor(data$SaleCondition)
    data$RoofStyle      <- encodeAsFactor(data$RoofStyle)
    data$RoofMatl       <- encodeAsFactor(data$RoofMatl)
    data$Exterior1st    <- encodeAsFactor(data$Exterior1st)
    data$Exterior2nd    <- encodeAsFactor(data$Exterior2nd)
    data$Foundation     <- encodeAsFactor(data$Foundation)
    #Foundation ad Ordinal -> TODO study how change the model using different values for Foundation
    #data$Foundation<-as.integer(revalue(data$Foundation, c('Slab'=1, 'BrkTil'=1, 'Stone'=1, 'CBlock'=1, 'Wood'=3, 'PConc'=3)))
    data$ExterQual      <- encodeAsOrdinal(data$ExterQual,Qualities)
    data$ExterCond      <- encodeAsOrdinal(data$ExterCond,Qualities)
    data$MasVnrType     <- encodeAsOrdinal(data$MasVnrType,Masonry, "None")
    data$BsmtQual       <- encodeAsOrdinal(data$BsmtQual,Qualities, "None")
    data$BsmtCond       <- encodeAsOrdinal(data$BsmtCond,Qualities, "None")
    data$BsmtExposure   <- encodeAsOrdinal(data$BsmtExposure,Exposure, "None")
    data$BsmtFinType1   <- encodeAsOrdinal(data$BsmtFinType1,FinType, "None")
    data$BsmtFinType2   <- encodeAsOrdinal(data$BsmtFinType2,FinType, "None")
    #Introducing new features Age, Remod, IsNew
    data$Age <- as.numeric(data$YrSold)-data$YearRemodAdd
	  data$Remod <- ifelse(data$YearBuilt==data$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
	  data$IsNew <- ifelse(data$YrSold==data$YearBuilt, 1, 0) #0=not new, 1=new
    data$YrSold <- as.factor(data$YrSold) #Numeric version is now not needed anymore
    data$BsmtFinSF1[is.na(data$BsmtFinSF1)] <-0
    data$BsmtFinSF2[is.na(data$BsmtFinSF2)] <-0
    data$BsmtUnfSF[is.na(data$BsmtUnfSF)] <-0
    data$TotalBsmtSF[is.na(data$TotalBsmtSF)] <-0
    data$MasVnrArea[is.na(data$MasVnrArea)] <-0
    #Introducing a new feature TotalSF
    data$TotalSF <- data$GrLivArea + data$TotalBsmtSF
    data <- data[-c(524, 1299),] #outliers
    data
}

handleLocations <- function(data){
    data$MSSubClass     <- encodeAsFactor(data$MSSubClass)
    data$MSZoning       <- encodeAsFactor(data$MSZoning)
    data$Street         <- encodeAsOrdinal(data$Street, AccessType)
    data$Alley          <- encodeAsFactor(data$Alley, "None")
    data$Neighborhood   <- encodeAsFactor(data$Neighborhood)
    data$Condition1     <- encodeAsFactor(data$Condition1)
    data$Condition2     <- encodeAsFactor(data$Condition2)
    data
}

handleLot <- function(data){
    data                <- getValidFrontages(data)
    data$LotShape       <- encodeAsOrdinal(data$LotShape, LotShape)
    data$LandContour    <- encodeAsFactor(data$LandContour)
    data$LotConfig      <- encodeAsFactor(data$LotConfig)
    data$LandSlope      <- encodeAsOrdinal(data$LandSlope, LandSlope)
    data
}

handleMisc <- function(data){
    data$Utilities      <- encodeAsOrdinal(data$Utilities, Utilities, "None")
    data$BldgType       <- encodeAsFactor(data$BldgType)
    data$HouseStyle     <- encodeAsFactor(data$HouseStyle)
    data$Heating        <- encodeAsFactor(data$Heating)
    data$HeatingQC      <- encodeAsOrdinal(data$HeatingQC, Qualities, 'None')
    data$CentralAir     <- encodeAsOrdinal(data$CentralAir, CentralAir, 'N')
    data$Electrical     <- encodeAsFactor(data$Electrical)
    data$FireplaceQu    <- encodeAsOrdinal(data$FireplaceQu, Qualities, "None")
    data                <- getValidMiscFeaturesAndVal(data)
    data$MiscFeature    <- encodeAsFactor(data$MiscFeature, "None")
    data
}

addFeatureBathrooms <- function(data){
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$TotBathRms <- data$BsmtFullBath + 0.5*data$BsmtHalfBath + data$FullBath + 0.5*data$HalfBath
    data$SalePrice <- prices
    data
}

addFeatureRecentGarage <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$RecentGarage[data$GarageYrBlt < 2000] <- 0
    data$RecentGarage[data$GarageYrBlt >= 2000] <- 1
    data$RecentGarage <- as.factor(data$RecentGarage)
    data$SalePrice <- prices
    data
}

addFeatureCarsXArea <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$GarageCarsTimesArea <- data$GarageCars * data$GarageArea
    data$SalePrice <- prices
    data
}

addFeatureRecentType <- function(data) {
    prices <- data$SalePrice
    data$SalePrice <- NULL
    data$GarageRecentType <- ifelse(data$GarageType=='BA' & data$RecentGarage==1, 1, 0)
    data$SalePrice <- prices
    data
}

getFactorData <- function(data) {
    newData <- data[, getFactorFields(data)]
    newData
}

getFactorFields <- function(data) {
    newData <- which(sapply(data, is.factor))
    names(newData)
}
