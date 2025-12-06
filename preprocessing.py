from fitparse import FitFile
import csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



fitfile = FitFile('ACTIVITY.fit')
k=0
D=set()
for record in fitfile.get_messages('record'):
  k=k+1
  for field in record:
    D.add(field.name)
#print(D)
cols=list(D)
df=pd.DataFrame(columns=list(D))
for d in df.columns:
  if d=="activity_type":
    df[d]=["o"]*k
  elif d=="timestamp":
    df[d]=pd.Timestamp('2025-01-01')
  else:
    df[d]=np.zeros(k)
df.describe()
k=0
for record in fitfile.get_messages('record'):
  for field in record:
    df.loc[k,field.name] = field.value
  k=k+1

#df=df.drop(columns=['unknown_87'])
#df=df.drop(columns=['unknown_138'])
#df=df.drop(columns=['unknown_137'])
#df=df.drop(columns=['unknown_136'])
#df=df.drop(columns=['unknown_135'])
#df=df.drop(columns=['unknown_134'])
#df=df.drop(columns=['unknown_107'])
#df=df.drop(columns=['unknown_140'])
#df=df.drop(columns=['unknown_144'])

cols=list(df.columns)
cols[cols.index('unknown_143')]='Avg Respiration Rate'
df.columns=cols

df['enhanced_altitude']=np.hstack((np.array([0]),np.diff(df.enhanced_altitude)))
relative = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds()/60
df["Time"]=relative
cols=['stance_time','Time','step_length','distance','heart_rate','enhanced_speed','enhanced_altitude','vertical_oscillation']
df=df[cols]
df.describe()

plt.plot( df['distance'],df['vertical_oscillation'])

df.to_csv("final_dataset.csv", index=False)