import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

columnsName = ['x1','x2','x3','x4','category']
df = pd.read_csv('iris.csv',usecols = [0,1,2,3,4])
df.columns = columnsName
df['class'] = df.category.replace(['Iris-setosa', 'Iris-versicolor'], [0.0,1.0])
df = df.head(100)

train1 = df[0:80]
train1=train1.reset_index(drop=True)
val1 = df[80:100]
val1= val1.reset_index(drop=True)

train2 = df[20:100]
train2 = train2.reset_index(drop=True)
val2 = df[0:20]
val2 = val2.reset_index(drop=True)

f = df[0:20]
l = df[40:100]
train3 = f.append(l)
train3 = train3.reset_index(drop=True)
val3 = df[20:40]
val3 = val3.reset_index(drop=True)

f = df[0:40]
l = df[60:100]
train4 = f.append(l)
train4 = train4.reset_index(drop=True)
val4 = df[40:60]
val4 = val4.reset_index(drop=True)

f = df[0:60]
l = df[80:100]
train5 = f.append(l)
train5 = train5.reset_index(drop=True)
val5 = df[60:80]
val5 = val5.reset_index(drop=True)

def train(df, weight, learningrate, epoch):
    dtheta = [0,0,0,0,0]
    actual = []
    predictions = []
    sigmoids = []
    accuracys = []
    errors = []
    n = 0

    while(n<epoch):
    
        for i in range (len(df)):

            result = weight[0]*df['x1'][i]+weight[1]*df['x2'][i]+weight[2]*df['x3'][i]+weight[3]*df['x4'][i]+weight[4]
            sigmoid = 1/(1+np.exp(-result))
            for j in range(0,len(dtheta)-1):
                dtheta[j] = 2*df.iloc[i,j]*(df['class'][i]-sigmoid)*(1-sigmoid)*sigmoid
            dtheta[4] = 2*(df['class'][i]-sigmoid)*(1-sigmoid)*sigmoid

            for x in range(len(weight)): 
                weight[x] = weight[x] + learningrate * dtheta[x]

            if sigmoid >= 0.5:
                prediction = 1.0
            else: 
                prediction = 0.0

            actual.append(df['class'][i])
            predictions.append(prediction)
            sigmoids.append(sigmoid)

            TP,FP,TN,FN = performancemeasure(actual,predictions)

            accuracy = (TP+TN)/(TP+FP+TN+FN)
            err = error(actual,sigmoids)

        n+=1
        accuracys.append(accuracy)
        errors.append(err)
        
    return(weight,accuracys,errors)

def validasi(df,weight,learningrate,epoch):
    actual = []
    predictions = []
    sigmoids = []
    accuracys = []
    errors = []
    n = 0
  
    while(n<epoch):
    
        for i in range (len(df)):

            result = weight[0]*df['x1'][i]+weight[1]*df['x2'][i]+weight[2]*df['x3'][i]+weight[3]*df['x4'][i]+weight[4]
            sigmoid = 1/(1+np.exp(-result))

            if sigmoid >= 0.5:
                prediction = 1.0
            else: 
                prediction = 0.0

            actual.append(df['class'][i])
            predictions.append(prediction)
            sigmoids.append(sigmoid)

            TP,FP,TN,FN = performancemeasure(actual,predictions)

            accuracy = (TP+TN)/(TP+FP+TN+FN)
            err = error(actual,sigmoids)
            
        n+=1
        accuracys.append(accuracy)
        errors.append(err)
        
    return(accuracys,errors)

def performancemeasure(actual, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(predictions)): 
        if actual[i]==predictions[i]==0:
            TP += 1
        if actual[i]==1 and actual[i]!=predictions[i]:
            FP += 1
        if actual[i]==predictions[i]==1:
            TN += 1
        if actual[i]==0 and actual[i]!=predictions[i]:
            FN += 1
    return(TP, FP, TN, FN)

def error(actual, predictions):
    error = 0.0
    for i in range(len(actual)):
        error += 1/2*(predictions[i] - actual[i])**2
    mean = error / len(actual)
    return mean
	
weight = [0.5,0.5,0.5,0.5,0.5]

weight1,accuracy1,error1 = train(train1,weight,0.1,300)
weight2,accuracy2,error2 = train(train2,weight,0.1,300)
weight3,accuracy3,error3 = train(train3,weight,0.1,300)
weight4,accuracy4,error4 = train(train4,weight,0.1,300)
weight5,accuracy5,error5 = train(train5,weight,0.1,300)

vaccuracy1,verror1 = validasi(val1,w1,0.1,300)
vaccuracy2,verror2 = validasi(val2,w2,0.1,300)
vaccuracy3,verror3 = validasi(val3,w3,0.1,300)
vaccuracy4,verror4 = validasi(val4,w4,0.1,300)
vaccuracy5,verror5 = validasi(val5,w5,0.1,300)

meanaccuracy = []
for i in range (len(accuracy1)):
    meanaccuracy.append((accuracy1[i]+accuracy2[i]+accuracy3[i]+accuracy4[i]+accuracy5[i])/5)

meanaccuracyv=[]
for i in range (len(vaccuracy1)):
    meanaccuracyv.append((vaccuracy1[i]+vaccuracy2[i]+vaccuracy3[i]+vaccuracy4[i]+vaccuracy5[i])/5)

meanerror = []
for i in range (len(error1)):
    meanerror.append((error1[i]+error2[i]+error3[i]+error4[i]+error5[i])/5)

meanerrorv = []
for i in range (len(verror1)):
    meanerrorv.append((verror1[i]+verror2[i]+verror3[i]+verror4[i]+verror5[i])/5)
		
x=plt.figure()
plt.suptitle('Grafik Akurasi dengan Learning Rate = 0.1')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(meanaccuracyv,'-ob')
plt.plot(meanaccuracy,'-om')
plt.gca().legend(('akurasi data validasi','akurasi data train'))
y=plt.figure()

y.suptitle('Grafik Error dengan Learning Rate = 0.1')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(meanerrorv,'-ob')
plt.plot(meanerror,'-om')  
plt.gca().legend(('error data validasi','error data train'))

weight = [0.5,0.5,0.5,0.5,0.5]

weight1,accuracy1,error1 = train(train1,weight,0.8,300)
weight2,accuracy2,error2 = train(train2,weight,0.8,300)
weight3,accuracy3,error3 = train(train3,weight,0.8,300)
weight4,accuracy4,error4 = train(train4,weight,0.8,300)
weight5,accuracy5,error5 = train(train5,weight,0.8,300)

vaccuracy1,verror1 = validasi(val1,weight1,0.8,300)
vaccuracy2,verror2 = validasi(val2,weight2,0.8,300)
vaccuracy3,verror3 = validasi(val3,weight3,0.8,300)
vaccuracy4,verror4 = validasi(val4,weight4,0.8,300)
vaccuracy5,verror5 = validasi(val5,weight5,0.8,300)

meanaccuracy = []
for i in range (len(accuracy1)):
    meanaccuracy.append((accuracy1[i]+accuracy2[i]+accuracy3[i]+accuracy4[i]+accuracy5[i])/5)
print(meanaccuracy)

meanaccuracyv=[]
for i in range (len(vaccuracy1)):
    meanaccuracyv.append((vaccuracy1[i]+vaccuracy2[i]+vaccuracy3[i]+vaccuracy4[i]+vaccuracy5[i])/5)
print(meanaccuracyv)

meanerror = []
for i in range (len(error1)):
    meanerror.append((error1[i]+error2[i]+error3[i]+error4[i]+error5[i])/5)
print(meanerror)

meanerrorv = []
for i in range (len(verror1)):
    meanerrorv.append((verror1[i]+verror2[i]+verror3[i]+verror4[i]+verror5[i])/5)
print(meanerrorv)

x=plt.figure()
plt.suptitle('Grafik Akurasi dengan Learning Rate = 0.8')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(meanaccuracyv,'-ob')
plt.plot(meanaccuracy,'-om')
plt.gca().legend(('akurasi data validasi','akurasi data train'))
y=plt.figure()

y.suptitle('Grafik Error dengan Learning Rate = 0.8')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(meanerrorv,'-ob')
plt.plot(meanerror,'-om')  
plt.gca().legend(('error data validasi','error data train'))