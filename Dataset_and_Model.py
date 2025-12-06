from Classes_and_Functions import NetLSTM
import numpy as np
import pandas as pd
import torch as th
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import itertools
import subprocess
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import json

script_path = "preprocessing.py"
subprocess.run(["python3", script_path], capture_output=False, text=False)
if th.cuda.is_available():
  device='cuda'
else:
  device='cpu'

# We set a random seed to repeat the project without the results changing.

np.random.seed(42)
th.manual_seed(42)

# We upload the dataset with pandas and we make the last transformations needed to use the model. Data are shown in the following chunks.


df=pd.read_csv("final_dataset.csv")
df.reset_index()
X=th.Tensor(df.iloc[:len(df.Time)-1,0:].values)
y=th.Tensor(df.iloc[1:,7].values)
x_min=X.min(dim=0, keepdim=True).values
x_max=X.max(dim=0, keepdim=True).values
X=(X - x_min) / (x_max - x_min)
y_min=y.min(dim=0, keepdim=True).values
y_max=y.max(dim=0, keepdim=True).values
y=(y - y_min) / (y_max - y_min)
X=X.to(device)
X.device
X=X.unsqueeze(1)
#print(X.shape)
y=y.to(device)

y=y.unsqueeze(1)
y=y.unsqueeze(2)

#print(y.shape)

x_final,x_test,y_final,y_test=train_test_split(X,y,test_size=0.33,train_size=0.67, shuffle=False)
x_train,x_val,y_train,y_val=train_test_split(x_final,y_final,test_size=0.33,train_size=0.67, shuffle=False)

x_train=x_train.to(device)
x_test=x_test.to(device)
x_val=x_val.to(device)
x_final=x_final.to(device)
y_train=y_train.to(device)
y_test=y_test.to(device)
y_val=y_val.to(device)
y_final=y_final.to(device)

# We now define a "toy problem" with batch size 20 and a little number of epochs to test the model programmed before.


hDim1=[(256,128),(128,64),(64,32)]
hDim2=[(256,128,64),(256,64,32),(128,64,32)]
epochs=100
hDim=hDim1
batch_sizes=[10,20,50,100]
lrs=[0.001,0.005,0.01]

table_of_results=dict()
with open("results.json", "r", encoding="utf-8") as f:
    table_of_results = json.load(f)
    table=dict()
    for i in table_of_results:
      table[float(i)]=table_of_results[i]
    table_of_results=table
for i in table_of_results:
  table_of_results[i][0]=tuple(table_of_results[i][0])
  table_of_results[i]=tuple(table_of_results[i])
#table_of_results=dict()
count=0
    
for hdim, batch_size,lr in itertools.product(hDim,batch_sizes,lrs):
    np.random.seed(42+count)
    th.manual_seed(42+count)
    count+=1
    net=NetLSTM(x_train.shape[2],hdim)
    net=net.to(device)
    if (hdim,batch_size,lr,net.strati) not in table_of_results.values():
        net.train(x_train,y_train,batch_size,0,epochs,lr)
        val_error=net.test(x_val,y_val)
        print(f"Strati: %d , batch size: %d, hidden dimensions: %d %d, lr %.3f validation error: %.5f"% (net.strati,batch_size,hdim[0],hdim[1],lr,val_error))
        table_of_results[val_error]=(hdim,batch_size,lr,net.strati)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(table_of_results, f, ensure_ascii=False, indent=4)
    else:
        for key in table_of_results:
          if table_of_results[key]==(hdim,batch_size,lr,net.strati):
            print(f"Strati: %d , batch size: %d, hidden dimensions: %d %d, lr %.3f validation error: %.5f"% (net.strati,batch_size,hdim[0],hdim[1],lr,key))
            break
hDim=hDim2


for hdim, batch_size,lr in itertools.product(hDim,batch_sizes,lrs):
    np.random.seed(42+count)
    th.manual_seed(42+count)
    count+=1
    net=NetLSTM(x_train.shape[2],hdim)
    net=net.to(device)
    if (hdim,batch_size,lr,net.strati) not in table_of_results.values():
        net.train(x_train,y_train,batch_size,0,epochs,lr)
        val_error=net.test(x_val,y_val,1)
        print(f"Strati: %d , batch size: %d, hidden dimensions: %d %d %d, lr %.3f validation error: %.5f"% (net.strati,batch_size,hdim[0],hdim[1],hdim[2],lr,val_error))
        table_of_results[val_error]=(hdim,batch_size,lr,net.strati)
        with open("results.json", "w", encoding="utf-8") as f:
          json.dump(table_of_results, f, ensure_ascii=False, indent=4)
    else:
        for key in table_of_results:
            if table_of_results[key]==(hdim,batch_size,lr,net.strati):
                print(f"Strati: %d , batch size: %d, hidden dimensions: %d %d %d, lr %.3f validation error: %.5f"% (net.strati,batch_size,hdim[0],hdim[1],hdim[2],lr,key))
                break

    


# We order the model's parameters on their validation error and we select the best configuration in order to retrain it.

win=list(table_of_results.keys())
win.sort()
B=[ (i,table_of_results[i]) for i in win]
best=B[0][1]


# We concatenate training and validation sets and we train on it the best model selected before. We then plot training loss' trend. 

strati=best[3]
hdim=best[0]
batch_size=best[1]

net=NetLSTM(x_final.shape[2],hdim)
lr=best[2]
a=net.train(x_final,y_final,batch_size,1,epochs,lr)

th.save(net.state_dict(), 'final_model.pth') 
print("Model saved successfully.")

# We evaluate the test error of the final model to understand its generalization capability.


test_error=net.test(x_test,y_test)
print(test_error)





