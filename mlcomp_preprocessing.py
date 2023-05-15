# -*- coding: utf-8 -*-
"""
Machine learning competition
step : data preprocessing
"""
# %% import librairies 

import os
import pandas as pd
import pickle
import numpy as np

# %% path and importing data

path1 = "E:/DOCUMENTS/CIVILE/BELGIQUE/MASTER_UCL/LSBA_2021/DATS2M/"

path2 = "BLOC_2/QUADRI_2/LINFO2262_MACHINE_LEARNING CLASSIFICATION/PROJECTS/ML_COMP/mywork/"

path_train = path1 + path2 + "ML-A5-2023_train"

data = pd.read_csv(path_train+"/batch_data.csv", index_col=0)

path_test = path1 + path2 + "ML-A5-2023_test"

datatest = pd.read_csv(path_test+"/batch_data.csv",  index_col=0)

del path1, path2

# del path_train
# %%
os.getcwd()

# %% data types

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

# %% numeric features and categorical one

# target dataframe

dataTarget = data[binary_var]

# numeric dataframe
dataNumeric = data[numeric_var]

# categorical dataframe
dataCat = data[cat_var]

# to use to impute missing data later
median_values = dataNumeric.median()
mode_values = dataCat.mode()

# %% detect missing values in features

def realMissingfeatures(dt, features):
    treatMissing = dt[features].isna().sum()
    realMissing = list()
    for i in range(len(treatMissing)) : 
        if treatMissing[i] != 0:
            realMissing.append(features[i])
    return realMissing, len(realMissing)

realMissing = realMissingfeatures(data, numeric_var)

# %% replace missing values in numeric features

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

dataNumericWithNoMissing, numeric_var, numeric_dropped = \
replace_with_median(dta = dataNumeric)

# del numeric_dropped, dataNumeric

dataNumericWithNoMissing.isna().values.any()

# %% replace missing values in categorical features

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

dataCatWithNoMissing.isna().values.any()

del dataCat

# %% median and mode values after features obtained

median_values_ = median_values[numeric_var]
# mode_values_ = mode_values[cat_var]


# %% handling the others dataframes

dir_list_train = os.listdir(path_train)[1:]

print("Files and directories in '", "ML-A5-2023_train", "' :")

# prints all files
print(dir_list_train)

# %% importing the probe data 

def import_data_probe(path = path_train):
    
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
# %% 
probedata, probedata_keys, batch_id = import_data_probe()

# %% importing one and performing analysis

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
            
    return pd.DataFrame(summaryStatistics, index=batch_id)
# %%
probeSummarydata = extracted_info_probe()

# %% the uninformative probe

mean_probe = probeSummarydata.mean()

def uninformative_probe(dta = probeSummarydata, meanProbe = mean_probe):
    """
    frop the features with mean 0 from probeSummarydata
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
# %% standardization of numeric features from cleaned dataToUse

from sklearn.preprocessing import StandardScaler

scalerNumericFeatures = StandardScaler().fit(dataNumericWithNoMissing)
scalerProbedata = StandardScaler().fit(probeDataWithNoUninformative)


dataNumericWithNoMissingScaled = scalerNumericFeatures.transform(
    dataNumericWithNoMissing)
probeDataWithNoUninformativeScaled = scalerProbedata.transform(
    probeDataWithNoUninformative)



# %% function to turn category modalities into 

def retrieveData(dta = dataCatWithNoMissing):
    
    """
    retrieve categorical data and remove the trailing numer the real numbers
    
    """
    
    colName = dta.columns
    indexVal = dataCatWithNoMissing.index
    
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

# %% mutual information computing 

from sklearn.feature_selection import mutual_info_classif as mutual_info

# numpyVersionOfDataNumeric = dataNumericWithNoMissingScaled.to_numpy()

# numpyVersionOfprobedata = probeDataWithNoUninformativeScaled.to_numpy()

mi_rsult1 =  mutual_info(dataNumericWithNoMissingScaled, dataTarget["target"])

mi_rsult2 =  mutual_info(probeDataWithNoUninformativeScaled, 
                         dataTarget["target"])

mi_rsult3 =  mutual_info(retrievedData, dataTarget["target"])


# %%
#len(mi_rsult) == numpyVersionOfDataNumeric.shape[1]

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


#del mi1_dataframe, mi2_dataframe, mi3_dataframe, dependenteFeatures1, \
 #   dependenteFeatures2, dependenteFeatures3
# %% features selection

corrNumeric = sortedMI[sortedMI["mutual_info"] > 0.10].index 
corrProbe = sortedMI2[sortedMI2["mutual_info"] > 0.05].index 
corrCat = sortedMI3[sortedMI3["mutual_info"] > 0.04].index
corrCatGood = sortedMI3.index[:4]

# %% creating scaler for test set

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


# %% joining dataframe together

dataToUse = correctNumericData.join([correctProbeData, correctCatData,
                                     dataTarget])

print(data.shape[0] == dataToUse.shape[0])

del sortedMI, sortedMI2, sortedMI3, 

# %% serialization of object 

# storing_neccessary_files = [numeric_var, cat_var, binary_var,
#                            median_values,
#                            mode_values,
#                            dataNumericWithNoMissing, dataCatWithNoMissing,
#                            dataTarget, dataToUse,
#                            probedata, probedata_keys, batch_id,
#                            scaleNumericFeatures, scaleProbe,
#                            probeDataWithNoUninformative,
#                            dataNumericWithNoMissingScaled, 
#                            probeDataWithNoUninformativeScaled]

with open("coorNumeric.pickled", "wb") as coorNumeric_file:
    pickle.dump(corrNumeric, coorNumeric_file, 
                protocol=pickle.HIGHEST_PROTOCOL)

with open("coorProbe.pickled", "wb") as coorProbe_file:
    pickle.dump(corrProbe, coorProbe_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("coorCat.pickled", "wb") as coorCat_file:
    pickle.dump(corrCat, coorCat_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("dataToUseFeatSelec.pickled", "wb") as dataToUse_file:
    pickle.dump(dataToUse, dataToUse_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("corrCatGood.pickled", "wb") as cCat:
    pickle.dump(corrCatGood, cCat, protocol=pickle.HIGHEST_PROTOCOL)


del coorNumeric_file, coorProbe_file, coorCat_file, dataToUse_file, cCat

# %% loading the pickled variable

import pickle

with open("coorNumeric.pickled", "rb") as numericFeature_file:
    corrNumeric = pickle.load(numericFeature_file)
    
with open("coorProbe.pickled", "rb") as probeFeature_file:
    corrProbe = pickle.load(probeFeature_file)
       
with open("coorCat.pickled", "rb") as categoricalFeature_file:
    corrCat = pickle.load(categoricalFeature_file)
    
with open("cCatGood.pickled", "rb") as cCatGood_file:
    corrCatGood = pickle.load(cCatGood_file)
    
with open("dataToUseFeatSelec.pickled", "rb") as dataToUse_file:
    dataToUse = pickle.load(dataToUse_file)
    
del dataToUse_file, categoricalFeature_file, probeFeature_file,
del numericFeature_file, cCatGood_file

# %% treating categorical features in dataToUse
from scipy import stats

pval = list()

for k in range(len(corrCat)):
    mee = pd.DataFrame(pd.crosstab(dataToUse[corrCat[0]], 
                                   dataToUse[corrCat[k]]))
    pval.append(stats.chi2_contingency(mee)[1])
    
pvalDataframe = pd.DataFrame(pval, index=corrCat, columns=["p_value"])

del pval

#pval2 = stats.fisher_exact(mee)[1]

# %% second mutual information to remove redondante features in categories

from sklearn.feature_selection import mutual_info_classif as mutual_info

dataToUseCat = dataToUse[corrCat]

dataToUseCatToNumpy = dataToUseCat.to_numpy()

mutual = list()

for cateFeat in corrCat: 
    mutual.append(mutual_info(dataToUseCatToNumpy, dataToUseCat[cateFeat]))
    
catMutualInforDataFrame = pd.DataFrame(mutual, index = corrCat, 
                                       columns=corrCat)

# features are so correlated that we may get some redundancy 

# %% working toward PCA

numericForPCA1 = dataToUse.select_dtypes(include=['number'])

numericForPCA = numericForPCA1[numericForPCA1.columns[:-1]]

del numericForPCA1

# %% treating numeric data through PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nbrComp = 8

pca = PCA(n_components=nbrComp).fit(numericForPCA)

colName = ["comp"+str(i) for i in range(1,nbrComp+1)]

indexVal = numericForPCA.index

dataToUseNumericReduced = pd.DataFrame(pca.transform(numericForPCA),
                                       index= indexVal, columns=colName)

sum(pca.explained_variance_ratio_)


# %% forming data from extraction

dataGoodCat = dataToUse[corrCatGood]

GreatData =  dataToUseNumericReduced.join([dataGoodCat, dataTarget])

dataGoodCat2 = dataToUse[corrCatGood[:2]]

GreatData2 = dataToUseNumericReduced.join([dataGoodCat2, dataTarget])

# %% register pca object for use in test set preprocessing

with open("pca.pickled", "wb") as pca_file:
    pickle.dump(pca, pca_file, 
                protocol=pickle.HIGHEST_PROTOCOL)

with open("dataToUseFeatSelec.pickled", "wb") as dataToUse_file:
    pickle.dump(dataToUse, dataToUse_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("dataToUseFeatExtract.pickled", "wb") as dataToUseExtr_file:
    pickle.dump(GreatData, dataToUseExtr_file, 
                protocol=pickle.HIGHEST_PROTOCOL)
    
with open("dataToUseFeatExtract2.pickled", "wb") as dataToUseExtr2_file:
    pickle.dump(GreatData2, dataToUseExtr2_file, 
                protocol=pickle.HIGHEST_PROTOCOL)
    
del pca_file, dataToUse_file, dataToUseExtr_file, dataToUseExtr2_file


# %% loading useful object back in memory

import pickle

with open("pca.pickled", "rb") as pca_file:
    pca = pickle.load(pca_file)
    
with open("dataToUseFeatSelec.pickled", "rb") as dataToUse_file:
    dataToUse = pickle.load(dataToUse_file)
       
with open("dataToUseFeatExtract.pickled", "rb") as greatData_file:
    greatData = pickle.load(greatData_file)
    
with open("dataToUseFeatExtract2.pickled", "rb") as greatData2_file:
    greatData2 = pickle.load(greatData2_file)
    
del pca_file, dataToUse_file, greatData_file, greatData2_file
    
# %% dataTest

# %%
import pickle

with open("dataTest2.pickled", "wb") as dataTest2_file:
    pickle.dump(dataTestGreat2, dataTest2_file, 
                protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open("dataTest.pickled", "wb") as dataTest_file:
    pickle.dump(dataTestGreat, dataTest_file, 
                protocol=pickle.HIGHEST_PROTOCOL)
    
# %%   
del  dataTest2_file, dataTest_file

    
    
    
    
    
    
    
    
    
    
    