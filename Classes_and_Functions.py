import torch as th
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import torch.utils.data as th_data
from tqdm import tqdm


class NetLSTM(nn.Module):
    def __init__(self,indim,hDim):
        super().__init__()
        self.strati=len(hDim)
        self.n_layers=len(hDim)
        define=[indim]+list(hDim)
        
        layer_list=list()
        for i in range(self.n_layers):
            layer_list.append(nn.LSTM(define[i],define[i+1],1))
        self.dense=nn.Linear(define[-1],1)
        self.layers=nn.ModuleList(layer_list)

    def forward(self,x,c_ts=[],h_ts=[]):
        if not c_ts:
            c_ts=[]
            h_ts=[]
            for i in range(self.n_layers):
                c_ts.append(th.zeros(1,self.layers[i].hidden_size,device=x.device.type))
                h_ts.append(th.zeros(1,self.layers[i].hidden_size,device=x.device.type))
        out=list()
        for xi in x:
            for j,f in enumerate(self.layers):
                xi,(h_t,c_t)=f(xi,(h_ts[j],c_ts[j]))
                c_ts[j]=c_t
                h_ts[j]=h_t
            xi=self.dense(xi)
            out.append(xi)
        out=th.stack(out,dim=0)
            
        return out.reshape(out.shape,1,1),c_ts,h_ts

    
    def test(self,x,y,plot=0):
        outputs, h_t,c_t=self(x)    
        y=y.reshape(outputs.shape).float()
        error=F.mse_loss(outputs,y).item()
        
        if plot:
            outputs=outputs.reshape(outputs.shape[0])
            os.makedirs("Immagini", exist_ok=True)
            labels=y.reshape(y.shape[0])
            outputs=[i.item() for i in outputs]
            labels=[i.item() for i in labels ]
            
            fig=plt.figure(figsize=(10,5))
            plt.plot(labels,color="green",label="True")
            plt.plot(outputs,label="Predicted")
            plt.legend()
            plt.savefig("Immagini/Results.png")
            plt.show()

        return error


    def train(self,x_train,y_train,batch_size,plot=0,epochs=100,lr=0.001):   
        
        criterion=nn.MSELoss()
        optimizer=optim.Adam(self.parameters(),lr=lr)
        losses=list()
        loader = th_data.DataLoader(th_data.TensorDataset(x_train, y_train), shuffle=False, batch_size=batch_size)
        
        for epoch in tqdm(range(epochs)):
            total_loss=0.0
            h_ts=[]
            for inputs, labels in loader:
                optimizer.zero_grad()
                if not h_ts:
                    outputs, c_ts,h_ts=self(inputs)
                else:
                    outputs, c_ts,h_ts=self(inputs,c_ts,h_ts)
                labels=labels.reshape(outputs.shape)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                th.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                total_loss+=loss.data.item()*inputs.shape[0]
                for j in range(self.n_layers):
                    c_ts[j]=c_ts[j].detach()
                    h_ts[j]=h_ts[j].detach()
            total_loss/=x_train.shape[0]
            #print(total_loss)
            losses.append(total_loss)

        if plot:
            os.makedirs("Immagini", exist_ok=True)

            fig, ax1=plt.subplots(figsize=(20,5))
            ax1.plot(losses)
            plt.savefig("Immagini/Losses.png")
            plt.show()
                
        return losses

        

   