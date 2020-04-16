######################################################################################################################################
# End to End Machine Learning Project
# Predict if a Flight Would Be On-Time for American Airlines
# By
# - Salma Alqahtani (ID Numer: 441813779) and Samar Alqahtani (ID Numer: 441814067)

# Keywords. Binary Classification; Predictive Model; Logistic Regression; Random Forest
######################################################################################################################################

######################################################################################################################################
# 1. Import Essentail Libraries # 
######################################################################################################################################

# To analyze this dataframe, the following packages are required :

# tables packages
library("dataframe.table")# package used for aggregation of large dataframesets 
library("dplyr") # package used for dataframe manipulation purposes
library("DT")  # Package used to customize tables

# plotlting and visualization packages
library("ggplot2") # package used for dataframe visualization purposes
library("ggthemes") # Package usedto apply themes to plots
library("maps") #package used for map visuaization purposes
library("leaflet")  #package used for visualization purposes
library("reshape2")
library("ggthemes")

# machine learning packages
library("caret")
library("randomForest")
library("e1071")
library("MASS")
library("ROCR")
library("pROC")
library("xgboost")

######################################################################################################################################
# 2. SELECTION OF dataframeSET # 
######################################################################################################################################




setwd("C:/Users/S O F T/Desktop/ml_project/")

origData <- read.csv2("C:/Users/S O F T/Desktop/ml_project/data/OnTimeFlightsJan2015.csv",
                      sep=",", header=TRUE, stringsAsFactors=FALSE)

# we read in in 469,968 rows of data
nrow(origData)

# That is a lot of rows to process so to speed thing up let's restrict dataframe to only flights
# between certain large airports
airports <-c('ATL','LAX', 'ORD', 'DFW', 'JFK', 'SFO', 'CLT', 'LAS', 'PHX')
origData <- subset(origData, DEST %in% airports & ORIGIN %in% airports)

# let's conserve the origData variable and add a new one called dataframe, that we will do all 
# the operations of cleaning an processing 
dataframe = origData

# just like that we are down to a more managable 32,716 rows
nrow(dataframe)

# print names of the features (attributes) : Column names 
names(dataframe)

# Dimensions of the data frame
dim(dataframe)

# Overview of the featuresi n the data (Figure 1 in report )
# Structure of the data frame
str(dataframe)

######################################################################################################################################
# 3. DESCRIPTIVE STATISTICS TO PERFORM EDA #
######################################################################################################################################

# Statistics summary of the data (Figure 2 in report)
# Summary of the data frame
summary(dataframe)

# summary of the target variable
summary(dataframe$ARR_DEL15)

######################################################################################################################################
# 4. VISUALIZATION TO PERFORM EDA #
######################################################################################################################################

# Convert all character variables to factors
dataframe <- dataframe %>% mutate_if(is.character, as.factor)

# Drop all rows where thye have NA values in the ARR_DEL15 variable
dataframe <- dataframe[!is.na(dataframe$ARR_DEL15) & 
                         dataframe$ARR_DEL15!="" & 
                         !is.na(dataframe$DEP_DEL15) & 
                         dataframe$DEP_DEL15!="",]
# dataframe <- dataframe %>% filter(!is.na(dataframe$ARR_DEL15)) // other alternative for dropping NA 

# Getting percentage of each class in ARR_DEL15 variable 
round(prop.table(table(dataframe$ARR_DEL15))*100,2)

# Proportion of On Time Delayed Flights (Figure 3 in report)
dataframe %>% ggplot(aes(x=ARR_DEL15, fill=ARR_DEL15)) + geom_histogram(stat="count") +
  scale_fill_manual(values=c("#9F2042","#AEA200"))

# Proportion of On Time Delayed flights grouped by airline (Figure 4 in report)
dataframe %>% ggplot(aes(x=ARR_DEL15,fill=OP_CARRIER)) + geom_histogram(stat="count", position = 'dodge')


# Plot for flight arrivals status depending on departure delay (Figure 5 in report)
dataframe %>% ggplot(aes(x=DAY_OF_WEEK, fill=ARR_DEL15)) + geom_histogram(stat = "count", position="dodge")+
  scale_fill_manual(values=c("#9F2042","#AEA200"))


dataframe %>% filter(ARR_DEL15 == "1.00") %>% ggplot() +
  geom_bar(mapping = aes(x = DEP_TIME_BLK, fill = DEP_TIME_BLK),position = "dodge") +
  xlab("Departure Time Block") +
  ylab("Number of Flights Delayed") +
  scale_fill_discrete(guide=guide_legend(title="Departure time block",ncol = 2,keywidth = 2)) +
  theme(axis.text.x = element_text(angle=90))

# Plot for number of flights delayed grouped by departure time block (Figure 6 )
dataframe %>% ggplot(aes(x=ARR_TIME, fill=ARR_DEL15)) + geom_histogram(position = "dodge", binwidth = 50) + theme_minimal() +
  scale_fill_manual(values=c("#9F2042","#AEA200"))


# he number of flights per airline 
avg_nflights <- length(dataframe$OP_CARRIER) / length(unique(dataframe$OP_CARRIER))
ggplot(dataframe, aes(x = OP_CARRIER)) +
  geom_bar() +
  geom_hline(yintercept = avg_nflights, col = "blue") + 
  coord_flip() + 
  theme_tufte() +
  labs( x = "Airline", y = "Number of flights ") +
  ggtitle("Number of flights per Airline", subtitle = "Blue vertical line: Average number of flights of the airlines")

# (Figure 7 in report)
# What days of the week are the most popular to fly? The next visualization shows 
# a barplot of the number of flights per day of the week.
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 1] = 'Monday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 2] = 'Tuesday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 3] = 'Wednesday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 4] = 'Thursday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 5] = 'Friday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 6] = 'Saturday'
dataframe$DAY_OF_WEEK[dataframe$DAY_OF_WEEK == 7] = 'Sunday'

avg_nflights_day <- length(dataframe$DAY_OF_WEEK) / length(unique(dataframe$DAY_OF_WEEK))
ggplot(dataframe, aes(x = factor(DAY_OF_WEEK), fill = OP_CARRIER)) +
  geom_bar() +
  geom_hline(yintercept = avg_nflights_day, col = "blue") + 
  theme_tufte() +
  coord_flip()+
  labs( x = "Day of the week", y = "Number of flights ")+
  ggtitle("Number of flights per Day of the week", subtitle = "Blue line represents Average of flights in any given day")

######################################################################################################################################
# 5. dataframe PREPARATION #
######################################################################################################################################

# Changing osme variable to numeric since we copuld liek to calculate some correlations
dataframe$DEP_DEL15 <- as.numeric(dataframe$DEP_DEL15)
dataframe$CANCELLED  <- as.numeric(dataframe$CANCELLED)
dataframe$DIVERTED <- as.numeric(dataframe$DIVERTED)
dataframe$DISTANCE <- as.numeric(dataframe$DISTANCE)

# Correlation between arrival and departure delay times
# cor(dataframe$ARR_DEL15, dataframe$DEP_DEL15)

# A perfect 1.  So "ORIGIN_AIRPORT_SEQ_ID" and "ORIGIN_AIRPORT_ID" are moving in lock step. Let's check  "ORIGIN_AIRPORT_SEQ_ID" 
# and "ORIGIN_AIRPORT_ID" are moving in lock step.
cor(dataframe[c("ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_AIRPORT_ID")])

# Another perfect 1.  So "DEST_AIRPORT_ID"and "DEST_AIRPORT_ID" are also moving in lock step.
cor(dataframe[c("DEST_AIRPORT_ID", "DEST_AIRPORT_ID")])

# Let's drop the columns "ORIGIN_AIRPORT_SEQ_ID" and "DEST_AIRPORT_ID" since they are not providing any new dataframe
dataframe$ORIGIN_AIRPORT_SEQ_ID <- NULL
dataframe$DEST_AIRPORT_SEQ_ID <- NULL

# OP_UNIQUE_CARRIER and OP_CARRIER also look related, actually they look like identical. 
# We can see if the are identical by filtering the rows to those we they are different. 
# R makes this easy, no loops to write.  All iteration is done for us.
mismatched <- dataframe[dataframe$OP_CARRIER != dataframe$OP_UNIQUE_CARRIER,]
nrow(mismatched)
# 0 mismatched, so OP_UNIQUE_CARRIER and OP_CARRIER"identical.  So let's rid of the UNIQUE_CARRIER column
dataframe$OP_CARRIER_AIRLINE_ID <- NULL

# Changing the format of a column and all of the dataframe for the row in that column is hard in some languages, 
# but simple in R. We just type in a simple command
dataframe$DISTANCE <- as.integer(dataframe$DISTANCE)
dataframe$CANCELLED <- as.integer(dataframe$CANCELLED)
dataframe$DIVERTED <- as.integer(dataframe$DIVERTED)

# Let's take the Arrival departure and delay fields.  Sometime algorithm perform better  when you change the fields into 
# factors which are like enumeration values in other languages. This allows the algorithm to use count of when a value is 
# a discrete value.
dataframe$ARR_DEL15 <- as.factor(dataframe$ARR_DEL15)
dataframe$DEP_DEL15 <-as.factor(dataframe$DEP_DEL15)

# Let also change some other columns factors
dataframe$DEST_AIRPORT_ID<- as.factor(dataframe$DEST_AIRPORT_ID)
dataframe$ORIGIN_AIRPORT_ID<- as.factor(dataframe$ORIGIN_AIRPORT_ID)
dataframe$DAY_OF_WEEK <- as.factor(dataframe$DAY_OF_WEEK )
dataframe$DEST<- as.factor(dataframe$DEST)
dataframe$ORIGIN<- as.factor(dataframe$ORIGIN)
dataframe$DEP_TIME_BLK <- as.factor(dataframe$DEP_TIME_BLK)
dataframe$OP_CARRIER <- as.factor(dataframe$OP_CARRIER)

#Redundant info
dataframe$ORIGIN_AIRPORT_ID = NULL
dataframe$AIR_TIME = NULL
dataframe$X = NULL
dataframe$OP_UNIQUE_CARRIER <- NULL
dataframe$OP_CARRIER_AIRLINE_ID <- NULL
dataframe$DEST_AIRPORT_ID <- NULL
dataframe$DEST_AIRPORT_SEQ_ID <- NULL
dataframe$ORIGIN_AIRPORT_SEQ_ID <- NULL


# let see how many arrival delayed vs non delayed flights.  We use tapply tosee how many time ARR_DEL15 is TRUE, and how many times 
# it is FALSE
# tapply(dataframe$ARR_DEL15, dataframe$ARR_DEL15, length)

# We should check how many departure delayed vs non delayed flights
(6460 / (25664 + 6460))
# The fact that we have a reasonable number of delays (6460 / (25664 + 6460)) = 0.201 ~ (20%) is important.  

# Drop "" level in ARR_DEL15
dataframe <- droplevels(dataframe[!dataframe$ARR_DEL15 == "",])

str(dataframe)
######################################################################################################################################
# 6. MODEL BUILDING: MACHINE LEARNING ALGORITHMS #
######################################################################################################################################

# Set random number seed for reproducability
set.seed(04102020)

# Set the columns we are going to use to train algorithm
featureCols <- c("ARR_DEL15", "DAY_OF_WEEK", "OP_CARRIER", "DEST","ORIGIN","DEP_TIME_BLK")

dataframe <- dataframe[!is.na(dataframe$ARR_DEL15) & dataframe$ARR_DEL15!="", ] 
dataframe <- dataframe[dataframe$ARR_DEL15!="", ] 


# Created filtered version of dataframe dataframeframe
dataframeFiltered <- dataframe[, featureCols]

# TRainning samples
inTrainRows <- createDataPartition(dataframeFiltered$ARR_DEL15, p=0.70, list=FALSE)

# check the row IDs
head(inTrainRows,10)
head(dataframeFiltered,3)

# Create the training dataframe frame
traindata<- dataframeFiltered[inTrainRows,]

# Create the testing dataframe frame.  Notice the prefix "-" 
testdata<- dataframeFiltered[-inTrainRows,]

# Check split: Should be 70%
nrow(traindata)/(nrow(testdata) + nrow(traindata))

# Should be 30%
nrow(testdata)/(nrow(testdata) + nrow(traindata))

#######################################################################################################################################
# 6.1 LOGISTIC REGRESSION
#######################################################################################################################################

# Fitting the model

logisticRegModel <- train(ARR_DEL15~ ., data=dataframeFiltered, 
                          method="glm",family="binomial", 
                          trControl=trainControl(method="cv", number=10, repeats=10))

logisticRegModel


######################################################################################################################################
# 6.2 RANDOM FOREST
#######################################################################################################################################
# We use the Random Forest algorithm which creates multiple decision trees and uses bagging to improve performance. Install the package - this only needs to be done once.  After the package is installed comment out this line unless you really want the latest version of the package to be downloaded and installed
# install.packages('randomForest')

rfModel <- randomForest(traindata[-1], 
                        traindata$ARR_DEL15, 
                        proximity=TRUE, 
                        importance=TRUE)
rfModel

######################################################################################################################################
# 7. MODEL EVALUATION
#######################################################################################################################################

# Logistic Regression
logRegPrediction <- predict(logisticRegModel, testdata)
logRegPrediction
# RandomForest
rfValidation <- predict(rfModel, testdata)
rfValidation

######################################################################################################################################
# 8. VISULIAZATION OF FINALS RESULTS
#######################################################################################################################################

# Get detailed statistics of prediction versus actual via Confusion Matrix

# Logistic Regression
logRegConfMat <- confusionMatrix(logRegPrediction, testdata[,"ARR_DEL15"])
logRegConfMat

# RandomForest
rfConfMat <- confusionMatrix(rfValidation, testdata[,"ARR_DEL15"])
rfConfMat

# CONFUSION MATRIX LG
table <- data.frame(confusionMatrix(logRegPrediction, testdata[,"ARR_DEL15"])$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

# CONFUSION MATRIX RF
table <- data.frame(confusionMatrix(rfValidation, testdata[,"ARR_DEL15"])$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

#################### END OF THE CODE ################