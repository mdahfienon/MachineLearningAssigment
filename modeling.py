# -*- coding: utf-8 -*-
"""
Created on Sat May  6 08:35:01 2023

@author: MATHIAS

@title Machine learning competition
step : modeling

"""
#%% import modules 
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %%

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# %% function to compute BCR

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

# %% loading back object in memory

with open("dataToUseFeatExtract2.pickled", "rb") as greatData2_file:
    greatData2 = pickle.load(greatData2_file)
    
# with open("dataTest.pickled", "rb") as test_file:
#    dataTest = pickle.load(test_file)
  
with open("dataTest2.pickled", "rb") as test2_file:
    dataTest2 = pickle.load(test2_file)
    
del test2_file, greatData2_file

# %% features dataset and target data

features = greatData2.columns[:-1]

target = greatData2.columns[-1]

tar_0 = greatData2[greatData2["target"] == 0].sample(n=5)

tar_1 = greatData2[greatData2["target"] == 1].sample(n=10)

dataForTest = pd.concat([tar_0,tar_1])

B_XTest = dataForTest[features]

B_yTest = dataForTest[target]

dataForTrain = greatData2.loc[~greatData2.index.isin(dataForTest.index)]

del tar_0, tar_1

# %% subsetting the dataframe to avoid overfitting

def get_balanced_data(data = dataForTrain, size = 46):
    
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
    
    sub_target_0 = target_0.sample(n=int(np.round(nbr_0*0.7)))
    
    sub_target_1 = target_1.sample(n=int(np.round(nbr_0*1.1)))
    
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

# %% competing model 

model_1 = SGDClassifier()
model_2 = DecisionTreeClassifier()
model_3 = KNeighborsClassifier()
model_4 = RandomForestClassifier()
# model_5 = SVC(C=0.0001)
model_6 = VotingClassifier([
                            ('SGD', model_1),
                            ('Tree', model_2),
                            ("KNN", model_3),
                            ('RForest', model_4),
                            # ('SVC', model_5)
                            ],
                           voting="hard")
# %% on whole unbalanced data

WX2_sample = dataForTrain[features]

Wy2_sample = dataForTrain[target]

WX2_train, WX2_test, Wy2_train, Wy2_test = \
    train_test_split(WX2_sample, Wy2_sample, test_size=0.3, shuffle=True)

for model in (model_1, model_2, model_3, model_4, model_6):
    model.fit(WX2_train, Wy2_train)
    print(model.__class__.__name__, bcr(model.predict(WX2_test),
                                        Wy2_test))
    

print(bcr(model_4.predict(B_XTest),
                                    B_yTest))
# %% on subset of balanced data


X2_train, X2_test, y2_train, y2_test, sampleData, \
    restData, XTest, YTest = get_balanced_data(size = 50)

# %%
for model in (model_1, model_2, model_3, model_4, model_6):
    model.fit(X2_train, y2_train)
    print(model.__class__.__name__, bcr(model.predict(X2_test),
                                        y2_test))


print(bcr(model_4.predict(XTest), YTest))

# %% 

print(bcr(model_4.predict(B_XTest), B_yTest))


# %% data undersampling for RF model

rsult_bcr = list()
rsult_bcr2 = list()
rsult_bcr3 = list()
model = RandomForestClassifier(criterion= 'entropy', max_features= 4)

for k in range(100):
    
    X2_train, X2_test, y2_train, y2_test, sampleData, \
        restData, XTest, YTest = get_balanced_data(size = 46)
        
    for j in range(100):
        model.fit(X2_train, y2_train)
        rsult_bcr.append(bcr(model.predict(X2_test), y2_test)) 
        rsult_bcr3.append(bcr(model.predict(XTest), YTest))
        rsult_bcr2.append(bcr(model.predict(B_XTest), B_yTest))                                                 

print("model_RF on X2_test", np.mean(rsult_bcr))
print("model_RF on BTest", np.mean(rsult_bcr2))
print("model_RF on XTest", np.mean(rsult_bcr3))

# %%

X2_train, X2_test, y2_train, y2_test, sampleData, \
    restData, XTest, YTest = get_balanced_data(size = 50)

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
    

print("X2",np.mean(bcrAnalysis), np.std(bcrAnalysis))
print("XGTest",np.mean(bcrGTest), np.std(bcrGTest))
print("B_Test",np.mean(bcrTest), np.std(bcrTest))

# %%
print(model_final.predict(dataTest2))



