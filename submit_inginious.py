###################################################################### 
##                                                                  ##
##                           MODULE IMPORTED                        ##
##                                                                  ##
###################################################################### 

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif as mutual_info
# from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

###################################################################### 
##                                                                  ##
##                           DATA IMPORTED                          ##
##                                                                  ##
###################################################################### 

data = pd.read_csv("ML-A5-2023_train/batch_data.csv", index_col=0)

datatest = pd.read_csv("ML-A5-2023_test/batch_data.csv", index_col=0)


###################################################################### 
##                                                                  ##
##                           CLEANING DATA                          ##
##                                                                  ##
###################################################################### 

# getting data types from 

def get_data_type(dta = data):
    
    """
    get the type of data
    """
    datatypes = dict(data.dtypes)
    
    numeric_var = [key for key in datatypes if datatypes[key] in 
                   ['float64', 'float32']]
    
    cat_var = [key for key in datatypes if datatypes[key] in 
                   ['object']]
    
    binary_var =  [key for key in datatypes if datatypes[key] in 
                   ["int32", 'int64']]

    return numeric_var, cat_var, binary_var


numeric_var, cat_var, binary_var = get_data_type()

# numeric features and categorical one

# target dataframe
dataTarget = data[binary_var]

# numeric dataframe
dataNumeric = data[numeric_var]

# categorical dataframe
dataCat = data[cat_var]

# to use to impute missing data later
median_values = dataNumeric.median()
mode_values = dataCat.mode()


# replace missing values in numeric features

def replace_with_median(dta, features = numeric_var, 
                        medianValues = median_values):
    
    """
    replace the missing value from the numeric features with the median
    
    """
    
    dt = dta.copy()
    dropped = list()
    retained = list()
    for feat in features:
        if dt[feat].isna().sum() > 100 :
            dt.drop(feat, axis=1, inplace = True)
            dropped.append(feat)
        elif dt[feat].isna().sum() != 0 :
            dt[feat].fillna(medianValues[feat], inplace=True)
            retained.append(feat)
        else :
            retained.append(feat)
        
    return dt, retained, dropped

dataNumericWithNoMissing, numeric_var, numeric_dropped = replace_with_median(dta = dataNumeric)

median_values_ = median_values[numeric_var]

# dataset of test data with numerical features
dataTestNumeric = datatest[numeric_var]


dataTestNumericWithNoMissing = replace_with_median(dta = dataTestNumeric,
                                                   features = numeric_var,
                                                   medianValues = median_values_)[0]

del numeric_dropped, dataNumeric

# replace missing values in categorical

def replace_with_mode(dta = data, features = cat_var, 
                      modeValues = mode_values):
    
    """
    replace the missing value from the categorical features with the mode
    
    """
    dt = dta.copy()
    retained = list()
    for feat in features:
        if dt[feat].isna().sum() > 100 :
            dt.drop(feat, axis=1, inplace = True)
        elif dt[feat].isna().sum() != 0 :
            #replace_with = dt[feat].mode()
            dt[feat].fillna(modeValues[feat][0], inplace=True)
            retained.append(feat)
        else :
            retained.append(feat)
            
    return dt, retained

dataCatWithNoMissing, cat_var = replace_with_mode(dta=dataCat)

mode_values_ = mode_values[cat_var]

# dataset of test data with categorical features
dataTestCat = datatest[cat_var]


dataTestCatWithNoMissing = replace_with_mode(dta = dataTestCat,
                                                   features = cat_var,
                                                   modeValues = mode_values_)[0]


del dataCat

# handling data from probes measures for train 

def import_data_probe(path):
    
    """
    get the probe data from path and create the dataframes    
    
    argument : path to directory
    
    return : 
        probedata : dictionary containing all the dataframe
        probedata_keys : probe name
        batch_id : id of batch 
    """
    
    dir_list = os.listdir(path)[1:]
    probedata = dict()
    
    for k in dir_list[:-1]:
        probedata["probe_" + k[27:-4]] = pd.read_csv(path+"/"+k)
    
    probedata_keys = list(probedata.keys())
    
    
    # getting the batch id
    batch_id = list(probedata[probedata_keys[1]]["batch_id"].unique())
    return probedata, probedata_keys, batch_id
 
probedata, probedata_keys, batch_id = import_data_probe(path = "ML-A5-2023_train")


def extracted_info_probe(probe = probedata_keys, datadict=probedata, 
                         batch = batch_id):
    """
    get summary data from probe datasets
    """

    summaryStatistics = dict()
    for p in probe:
        minstat = list()
        meanstat = list()
        maxstat = list()
        for l in batch:
            dataB = datadict[p]
            datasub = dataB[dataB["batch_id"]==l]["value_approx"]
            minstat.append(datasub.min())
            meanstat.append(datasub.mean())
            maxstat.append(datasub.max())
        summaryStatistics[p+"_min"] = minstat
        summaryStatistics[p+"_mean"] = meanstat
        summaryStatistics[p+"_max"] = maxstat
            
    return pd.DataFrame(summaryStatistics, index=batch)

# extracting summaries from probes of train  
probeSummarydata = extracted_info_probe()

# handling data from probes measures for test
probedataTest, probedata_keysTest, batch_idTest = import_data_probe(path = "ML-A5-2023_test")

# extracting summaries from probes of test  
probeSummarydataTest = extracted_info_probe(probe = probedata_keysTest,
                                            datadict = probedataTest,
                                            batch = batch_idTest)

# removing the uninformative probe data from train 

mean_probe = probeSummarydata.mean()

def uninformative_probe(dta = probeSummarydata, meanProbe = mean_probe):
    """
    drop the features with mean 0 from probeSummarydata
    """
    dt = dta.copy()
    dropped = list()
    for feat in dta.columns :
        if meanProbe[feat] == 0:
            dt.drop(feat, axis=1, inplace = True)
            dropped.append(feat)
    return dt, dropped

probeDataWithNoUninformative, dropfeatures = uninformative_probe()

del dropfeatures, probeSummarydata

# removing the uninformative probe data from test

mean_probe_test = probeSummarydataTest.mean()

probeDataTestWithNoUninformative = uninformative_probe(dta =probeSummarydataTest,
                                                       meanProbe = mean_probe_test)[0]


# scaling the train data 

scalerNumericFeatures = StandardScaler().fit(dataNumericWithNoMissing)
scalerProbedata = StandardScaler().fit(probeDataWithNoUninformative)


dataNumericWithNoMissingScaled = scalerNumericFeatures.transform(
    dataNumericWithNoMissing)
probeDataWithNoUninformativeScaled = scalerProbedata.transform(
    probeDataWithNoUninformative)

# scaling the test data using the scaler of the train set 

dataTestNumericWithNoMissingScaled = scalerNumericFeatures.transform(dataTestNumericWithNoMissing)

probeDataTestWithNoUninformativeScaled = scalerProbedata.transform(probeDataTestWithNoUninformative)


# first version of cleaned test data

dataTestNumericframe = pd.DataFrame(dataTestNumericWithNoMissingScaled,
                                      index = dataTestNumericWithNoMissing.index,
                                      columns=\
                                          dataTestNumericWithNoMissing.columns)

probeDataTestScaled = pd.DataFrame(probeDataTestWithNoUninformativeScaled,
                                index = probeDataTestWithNoUninformative.index,
                                columns=probeDataTestWithNoUninformative.columns)


dataTestNumericToUse_ = dataTestNumericframe.join(probeDataTestScaled)
# function to turn category modalities into 

def retrieveData(dta = dataCatWithNoMissing):
    
    """
    retrieve categorical data and remove the trailing numer the real numbers
    
    """
    
    colName = dta.columns
    indexVal = dta.index
    
    numpyVersionOfCatData = dta.to_numpy()
    
    retrievedData =  list()
    for l in numpyVersionOfCatData:
        interdata = list()
        for h in l:
            interdata.append(int(h[-1]))
        retrievedData.append(np.array(interdata))
        
    retrievedData = np.array(retrievedData)
    pandasData = pd.DataFrame(retrievedData, columns=colName, 
                              index=indexVal, dtype= 'category')
        
    return retrievedData, pandasData

retrievedData, retrieveDataPandas = retrieveData()

retrievedDataTest = retrieveData(dta = dataTestCatWithNoMissing)[1]


# computing mutual information between numeric features and the target 
mi_rsult1 =  mutual_info(dataNumericWithNoMissingScaled, dataTarget["target"])

mi_rsult2 =  mutual_info(probeDataWithNoUninformativeScaled, 
                         dataTarget["target"])

mi_rsult3 =  mutual_info(retrievedData, dataTarget["target"])


mi1_dataframe = pd.DataFrame(mi_rsult1, 
                             index=dataNumericWithNoMissing.columns,
                             columns=["mutual_info"])

dependenteFeatures1 = mi1_dataframe[mi1_dataframe["mutual_info"] !=0]

sortedMI = dependenteFeatures1.sort_values(by="mutual_info", ascending=False)

mi2_dataframe = pd.DataFrame(mi_rsult2, 
                             index=probeDataWithNoUninformative.columns,
                             columns=["mutual_info"])

dependenteFeatures2 = mi2_dataframe[mi2_dataframe["mutual_info"] !=0]

sortedMI2 = dependenteFeatures2.sort_values(by="mutual_info", ascending=False)

mi3_dataframe = pd.DataFrame(mi_rsult3, 
                             index=dataCatWithNoMissing.columns, 
                             columns=["mutual_info"])

dependenteFeatures3 = mi3_dataframe[mi3_dataframe["mutual_info"] !=0]

sortedMI3 = dependenteFeatures3.sort_values(by="mutual_info", ascending=False)


del mi1_dataframe, mi2_dataframe, mi3_dataframe, dependenteFeatures1, \
    dependenteFeatures2, dependenteFeatures3

corrNumeric = sortedMI[sortedMI["mutual_info"] > 0.10].index 
corrProbe = sortedMI2[sortedMI2["mutual_info"] > 0.05].index 
corrCat = sortedMI3[sortedMI3["mutual_info"] > 0.04].index
corrCatGood = sortedMI3.index[:4]


dataNumericScaledFrame = pd.DataFrame(dataNumericWithNoMissingScaled,
                                      index = dataNumericWithNoMissing.index,
                                      columns=\
                                          dataNumericWithNoMissing.columns)

dataProbeScaledFrame = pd.DataFrame(probeDataWithNoUninformativeScaled,
                                index = probeDataWithNoUninformative.index,
                                columns=probeDataWithNoUninformative.columns)


correctNumericData = dataNumericScaledFrame[corrNumeric]
correctCatData = retrieveDataPandas[corrCatGood]
correctProbeData = dataProbeScaledFrame[corrProbe]

dataTestNumericToUse = dataTestNumericToUse_[corrNumeric].join(dataTestNumericToUse_[corrProbe])

# joining dataframe together

dataToUse = correctNumericData.join([correctProbeData, correctCatData,
                                     dataTarget])

print(data.shape[0] == dataToUse.shape[0])

del sortedMI, sortedMI2, sortedMI3, 

# working toward PCA

numericForPCA1 = dataToUse.select_dtypes(include=['number'])

numericForPCA = numericForPCA1[numericForPCA1.columns[:-1]]

del numericForPCA1

#  treating numeric data through PCA

nbrComp = 8

pca = PCA(n_components=nbrComp).fit(numericForPCA)

colName = ["comp"+str(i) for i in range(1,nbrComp+1)]

indexVal = numericForPCA.index

dataToUseNumericReduced = pd.DataFrame(pca.transform(numericForPCA),
                                       index= indexVal, columns=colName)

# sum(pca.explained_variance_ratio_)

dataTestNumericReduced = pd.DataFrame(pca.transform(dataTestNumericToUse),
                                       index= dataTestNumericToUse.index,
                                       columns=colName)


##############  forming data from all features engineering ################

####  train data 

# categories 

dataGoodCat = dataToUse[corrCatGood]

dataGoodCat2 = dataToUse[corrCatGood[:2]]

# joined version of train data


greatData2 = dataToUseNumericReduced.join([dataGoodCat2, dataTarget])

####  test data 

# categories 

dataTestCatReduced2 = retrievedDataTest[corrCatGood[:2]]


# joined version of test data


dataTestGreat2 = dataTestNumericReduced.join(dataTestCatReduced2)


print( dataTestGreat2.columns == greatData2.columns[:-1], sep="\n")


###################################################################### 
##                                                                  ##
##                           MODELLING                              ##
##                                                                  ##
###################################################################### 

# function to compute BCR
def bcr(y_predict, y_true):
    """
    this function compute the BCR = 0.5*(TP/(TP+FN) + TN/(FP+TN))
    """
    
    confusionMatrix = pd.crosstab(index = y_predict, columns=y_true)
    
    TP = confusionMatrix[1][1]
    FN = confusionMatrix[1][0]
    TN = confusionMatrix[0][0]
    FP = confusionMatrix[0][1]
    
    return 0.5 *((TP/(TP+FN))+(TN/(FP+TN)))

features = greatData2.columns[:-1]

target = greatData2.columns[-1]

tar_0 = greatData2[greatData2["target"] == 0].sample(n=5)

tar_1 = greatData2[greatData2["target"] == 1].sample(n=10)

dataForTest = pd.concat([tar_0,tar_1])

B_XTest = dataForTest[features]

B_yTest = dataForTest[target]

dataForTrain = greatData2.loc[~greatData2.index.isin(dataForTest.index)]

del tar_0, tar_1


# function to get balanced data

def get_balanced_data(data = dataForTrain, size = 46, perc = 0.9):
    
    """
    produce balanced data based the data provided as input
    """
    dta = data.copy()
    
    features = dta.columns[:-1]

    target = dta.columns[-1]

    target_0 =  dta[dta["target"] == 0]
    
    if size <= target_0.shape[0]:
        nbr_0 = size
    else : nbr_0 = target_0.shape[0] 

    target_1 = dta[dta["target"] == 1]
    
    sub_target_0 = target_0.sample(n=int(np.round(nbr_0*perc)))
    
    sub_target_1 = target_1.sample(n=int(np.round(nbr_0*2.5*perc)))
    
    rest_0 = target_0.loc[~target_0.index.isin(sub_target_0.index)]
    
    rest_1 = target_1.loc[~target_1.index.isin(sub_target_1.index)]
    
    sampleData = pd.concat([sub_target_0,sub_target_1])
    
    restData = pd.concat([rest_0,rest_1])
    
    X2_sample = sampleData[features]
    
    y2_sample = sampleData[target]
    
    XTest, YTest = restData[features], restData[target]
    
    X2_train, X2_test, y2_train, y2_test = \
        train_test_split(X2_sample, y2_sample, test_size=0.3, 
                         shuffle=True, random_state= size)
    
    
    return X2_train, X2_test, y2_train, \
        y2_test, sampleData, restData, XTest, YTest

X2_train, X2_test, y2_train, y2_test, sampleData, \
    restData, XTest, YTest = get_balanced_data(size = 50)

############### Modeling to get the mean BCR ###########################

model_final = RandomForestClassifier(criterion= 'entropy', 
                                     max_features= 4)

X2_sample = sampleData[features]

y2_sample = sampleData[target]

bcrAnalysis = list()

bcrTest = list()

bcrGTest = list()

for h in range(100):

    X2_train, X2_test, y2_train, y2_test = \
        train_test_split(X2_sample, y2_sample, test_size=0.3, shuffle=True)
        
    model_final.fit(X2_train, y2_train)
    
    bcrAnalysis.append(bcr(model_final.predict(X2_test), y2_test))
    
    bcrTest.append(bcr(model_final.predict(B_XTest), B_yTest))
    
    bcrGTest.append(bcr(model_final.predict(XTest), YTest))
    

print("X2", np.mean(bcrAnalysis), np.std(bcrAnalysis))
print("XGTest", np.mean(bcrGTest), np.std(bcrGTest))
print("B_Test", np.mean(bcrTest), np.std(bcrTest))

###################################################################### 
##                                                                  ##
##                 FINAL MODEL AND PREDICTION                       ##
##                                                                  ##
###################################################################### 


model_final_1 = RandomForestClassifier(criterion= 'entropy', 
                                     max_features= 4)

X2_train, X2_test, y2_train, y2_test, sampleData, \
    restData, XTest, YTest = get_balanced_data(size = 55, perc=1)

model_final_1.fit(X2_train, y2_train)


print(bcr(model_final_1.predict(B_XTest), B_yTest))


# prediction file for test set

my_prediction = model_final_1.predict(dataTestGreat2)



pd.DataFrame(my_prediction, index=dataTestGreat2.index, 
                         columns=["Prediction"]).to_csv("prediction.csv", 
                                                        index_label="")



