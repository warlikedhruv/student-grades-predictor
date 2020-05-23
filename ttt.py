import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle as pk

train = pd.read_csv("marks.csv", sep=";")
test = pd.read_csv("marks.csv", sep=";")



train.isna().head()
test.isna().head()


passed=[]
for i in train.index:
    if  train['G3'][i]  > 8:
        passed.append(1)
    else:
        passed.append(0)


train['passed'] = passed
test['passed']= passed



train = train.drop(['school','address','famsize','Pstatus','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','famsup','paid','famrel','Dalc','Walc'], axis=1)
test = test.drop(['school','address','famsize','Pstatus','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','famsup','paid','famrel','Dalc','Walc'], axis=1)
train.info()


labelEncoder = LabelEncoder()
labelEncoder.fit(train['sex'])
labelEncoder.fit(test['sex'])
train['sex'] = labelEncoder.transform(train['sex'])
test['sex'] = labelEncoder.transform(test['sex'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['romantic'])
labelEncoder.fit(test['romantic'])
train['romantic'] = labelEncoder.transform(train['romantic'])
test['romantic'] = labelEncoder.transform(test['romantic'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['activities'])
labelEncoder.fit(test['activities'])
train['activities'] = labelEncoder.transform(train['activities'])
test['activities'] = labelEncoder.transform(test['activities'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['internet'])
labelEncoder.fit(test['internet'])
train['internet'] = labelEncoder.transform(train['internet'])
test['internet'] = labelEncoder.transform(test['internet'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['higher'])
labelEncoder.fit(test['higher'])
train['higher'] = labelEncoder.transform(train['higher'])
test['higher'] = labelEncoder.transform(test['higher'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['nursery'])
labelEncoder.fit(test['nursery'])
train['nursery'] = labelEncoder.transform(train['nursery'])
test['nursery'] = labelEncoder.transform(test['nursery'])

labelEncoder = LabelEncoder()
labelEncoder.fit(train['schoolsup'])
labelEncoder.fit(test['schoolsup'])
train['schoolsup'] = labelEncoder.transform(train['schoolsup'])
test['schoolsup'] = labelEncoder.transform(test['schoolsup'])

test.info()
print("train")
train.info()


demo_train = train.drop(['passed'], axis=1)
demo_train.info()
X = np.array(demo_train)
y = np.array(train['passed'])

with open('marks_kmeans', 'rb') as file:
    kmeans = pk.load(file)
#kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
#    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
#    random_state=None, tol=0.0001, verbose=0)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
print(X)



def predict_kmeans():
    correct = 0
    passed =0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if (prediction[0] == 1):
            passed += 1
        if prediction[0] == y[i]:
            correct += 1

    print("acc")
    print((correct/len(X) * 100))
    print((passed/len(X)*100))
    return [(correct/len(X) * 100), (passed/len(X)*100)]

list = predict_kmeans()

