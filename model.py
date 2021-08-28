import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#import os
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

plt.style.use('ggplot')
#%matplotlib inline

msd_data = pd.read_excel('MSD_Features.xls')
#msd_data
#msd_data.columns

bb_data = pd.read_excel('BillBoard_Features.xls')
#bb_data
#msd_data.isna().sum()
#bb_data.isna().sum()

msd_data['ID'] = msd_data[[4,5]].apply(lambda x: ' '.join(x),axis=1)
# Combining both columns of 4 and 5 to make a new column ID 

msd_data.drop(([4,5]),axis=1,inplace=True)
# Dropping those 2 as they are not needed anymore

msd_data = msd_data.rename({0:'Artist',1:'Album',2:'Track',3:'Year'},axis=1)
# Renaming the columns acc to basic sense

#msd_data.info()

bb_data.rename(columns={'SpotifyID':'ID'},inplace=True)
#bb_data
# changing the SpotifyID column name to ID so that it'll be easier to concat the two DFs
#bb_data.info()

msd_bb = pd.concat([msd_data,bb_data],axis=0,ignore_index=True)

#msd_bb
#msd_bb.info()
#msd_bb.isna().sum()
msd_bb=msd_bb.drop(['Album','Year'],axis=1)

#msd_bb['mode'].unique()
# We have this value of -999 that makes no sense in this column

msd_bb.replace(-999,np.nan,inplace=True)
#msd_bb['mode'].unique()
#msd_bb.isna().sum()

#msd_bb['danceability'].nunique()

msd_bb.dropna(inplace=True)
#msd_bb.isna().sum()
#msd_bb.shape

msd_bb=msd_bb.drop(['Track','Artist','ID'],axis=1)

#msd_bb
msd_bb = msd_bb.drop_duplicates()

#sns.countplot(x='mode', data=msd_bb,palette='hls')
#plt.show()
# data is imbalanced, will have to sample it. But will only do it after train test split



x_features = list(msd_bb.columns)
#x_features
x_features.remove('mode')


encoded_data = pd.get_dummies(msd_bb[x_features],drop_first=True)
#encoded_data

y = msd_bb['mode']
x = encoded_data

#msd_bb['mode'].shape
#encoded_data.shape

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=2000)

log_reg.fit(x_train,y_train)

y_pred = log_reg.predict(x_test)

from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)