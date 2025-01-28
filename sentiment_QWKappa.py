from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

#----------If I get None from LLM Output because of output format error I count it as I missed it--------- 

#Percentage for true predictions
human_raw = []
pred_raw = []
#Put if output from llm is string like "Positive"

for idx,p in enumerate(pred_raw):
    if p == "Positive":
        pred_raw[idx] = "1"
    elif p == 'Negative':
        pred_raw[idx] = "-1"
    elif p == "Neutral":
        pred_raw[idx] = "0"

for idx,p in enumerate(human_raw):
    if p == "Positive":
        human_raw[idx] = "1"
    elif p == 'Negative':
        human_raw[idx] = "-1"
    elif p == "Neutral":
        human_raw[idx] = "0"

print("len human_raw:")
print(len(human_raw)) 
print("human_raw:")       
print(human_raw)
print("len pred_raw:")
print(len(pred_raw))
print("pred_raw:")
print(pred_raw)

pred = []
human = []

for idx,p in enumerate(pred_raw):    
    
    if p != "NaN" and p != None:
        pred.append(p)
        human.append(human_raw[idx])
 
    


print("len human:")
print(len(human)) 
print("human:")       
print(human)
print("len pred:")
print(len(pred))
print("pred:")
print(pred)

#convert human from str to int
for idx, i in enumerate(human):
    human[idx] = int(i)
#print(human)


for idx, i in enumerate(pred):
   pred[idx] = int(i)        
#print(pred)


#ACCURACY, PRECISION, RECALL
perf = []
for x,y in zip(human, pred):
    if x == y:
        perf.append(1)
    else:
        perf.append(0)

metric = float(sum(perf) / len(perf))     
print(metric)
print(len(perf))


N=3

# label 2 is NONE type
O = confusion_matrix(human, pred, labels=[1,0,-1])
w = np.zeros((N,N))

print(O)

for i in range(len(w)):
    for j in range(len(w)):
        w[i][j] = float(((i-j)**2)/(N-1)**2) #as per formula, for this competition, N=6

print(w)

act_hist=np.zeros([N])
for item in human: 
    act_hist[item]+=1
    
pred_hist=np.zeros([N])
for item in pred: 
    pred_hist[item]+=1   

print(f'Actuals value counts:{act_hist}, Prediction value counts:{pred_hist}')   

E = np.outer(act_hist, pred_hist)
print(E)

E = E/E.sum()
print(E.sum())

O = O/O.sum()
print(O.sum())

print(E)
print(O)

num=0
den=0
for i in range(len(w)):
    for j in range(len(w)):
        num+=w[i][j]*O[i][j]
        den+=w[i][j]*E[i][j]
 
weighted_kappa = (1 - (num/den)); weighted_kappa

print(weighted_kappa)

