import subprocess
import pandas as pd
import numpy as np
from perceptron import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

subprocess.call(["bash","link_download.sh"]) #Download all necessary files from internet
data_clinical=pd.read_csv('clinical_data', index_col=0, sep='\t')

#data_clinical=pd.read_csv('clinical_data',index_col=0,sep='\t')
y = data_clinical['gender']


#print list(data_clinical.columns.values)
#DF = data_clinical[['_EVENT', '_INTEGRATION', '_OS','_OS_IND','_TIME_TO_EVENT', 'acute_myeloid_leukemia_calgb_cytogenetics_risk_category','days_to_death','gender','history_of_neoadjuvant_treatment','age_at_initial_pathologic_diagnosis','vital_status']]
#print list(data_clinical.columns.values)
#print DF
data_gen=pd.read_csv('genomicMatrix', index_col=0, sep='\t')

#DF=DF.sort(['gender'])
sd = data_gen.std(1)
sd.sort(ascending=False)
genes = list(sd[0:15000].index)

train_data, test_data = train_test_split(data_gen.ix[genes].T, train_size=0.8, random_state=0)
train_y = y.ix[train_data.index]
train_y = np.where(train_y == 'MALE', -1, 1)

test_y = y.ix[test_data.index]
test_y = np.where(test_y == 'MALE', -1, 1)

ppn = Perceptron(epochs=50, eta=0.1)
ppn.fit(train_data.values, train_y)
pred = ppn.predict(test_data)
print pred
print 'performance', np.sum(pred==test_y)*1.0/test_y.size*100.0