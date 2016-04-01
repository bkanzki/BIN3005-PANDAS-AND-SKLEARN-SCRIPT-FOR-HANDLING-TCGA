import subprocess
import pandas as pd
import numpy as np
from perceptron import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.tools as tls
tls.set_credentials_file(username='your_plotly_name', api_key='your_plotly_web_key')



############################################################################################################################################################################################################
# execute script bash
############################################################################################################################################################################################################

subprocess.call(["bash","link_download.sh"]) #Download all necessary files from internet


############################################################################################################################################################################################################
#Read newly downloaded files
############################################################################################################################################################################################################

data_clinical=pd.read_csv('clinical_data',sep='\t') #strore first file
#data_clinical=pd.read_csv('clinical_data',index_col=0,sep='\t')
y = data_clinical['gender']         #select gender column


#print list(data_clinical.columns.values)
#select certain columns 
DF = data_clinical[['_EVENT', '_INTEGRATION', '_OS', '_OS_IND','_TIME_TO_EVENT', 'acute_myeloid_leukemia_calgb_cytogenetics_risk_category','days_to_death','gender','history_of_neoadjuvant_treatment','age_at_initial_pathologic_diagnosis','vital_status']]
#print list(data_clinical.columns.values)
#print DF

#get genomic info
data_gen=pd.read_table('genomicMatrix')
DF=DF.sort(['gender'])

#data_gen=pd.read_table('genomicMatrix',index_col=0)

#DF = DF.set_index(['gender'])
#print(len(DF.loc['MALE']))

#transposition of matrix 
transpose=data_gen.set_index('sample').transpose()
transpose.to_csv('result.csv', sep=',') #store new transposed matrix
data_gen=pd.read_csv('result.csv')      # open the new file
data_gen= data_gen.rename(columns = {'Unnamed: 0': 'sampleID'}) #rename columns

#DF['event']=np.where(DF._EVENT==0,'0','1')
#DF['PCA_ANALYSIS']=DF.history_of_neoadjuvant_treatment
#DF['PCA_ANALYSIS_1']= DF.history_of_neoadjuvant_treatment+'_'+DF.vital_status
DF['PCA_ANALYSIS_2']= DF.gender+'_'+DF.history_of_neoadjuvant_treatment+'_'+DF.vital_status
#DF['PCA_ANALYSIS_3']= DF.gender+'_'+DF.acute_myeloid_leukemia_calgb_cytogenetics_risk_category+'_'+DF.event
#DF1=pd.concat([DF.PCA_ANALYSIS,data_clinical.sampleID],axis=1)
#DF2=pd.concat([DF.PCA_ANALYSIS_1,data_clinical.sampleID],axis=1)
DF3=pd.concat([DF.PCA_ANALYSIS_2,data_clinical.sampleID],axis=1)
#result = pd.concat([data_gen,DF1], axis=1)
#result=result.sort(['PCA_ANALYSIS'])
#result2 = pd.concat([data_gen,DF2], axis=1)
result3 = pd.concat([data_gen,DF3], axis=1, join_axes=[data_gen.index])

#result2=result2.sort(['PCA_ANALYSIS_1'])
result3=result3.sort(['PCA_ANALYSIS_2'])
#result.to_csv('PCA1.csv', sep=',')
#del data_gen['sampleID']
#del result['sampleID']
#del result['PCA_ANALYSIS']
#del result2['sampleID']
#del result2['PCA_ANALYSIS_1']
del result3['sampleID']
del result3['PCA_ANALYSIS_2']
#result.to_csv('PCA1.csv', sep=',')
#print result.PCA_ANALYSIS
#result.dropna(how="all", inplace=True)
result3.dropna(how="all", inplace=True)
#result3.dropna(how="all", inplace=True)
#print result3.PCA_ANALYSIS_2
#result3 = result3.set_index(['PCA_ANALYSIS_2'])

'''Uncomment this to know where each parameter stops

print (len(result3))
print(len(result3.loc['FEMALE_Yes_DECEASED']),'With Tx_alive')
print(len(result3.loc['FEMALE_No_DECEASED']),'Without Tx_alive')
print(len(result3.loc['FEMALE_Yes_LIVING']),'With Tx_alive')
print(len(result3.loc['FEMALE_No_LIVING']),'Without Tx_alive')
print(len(result3.loc['MALE_Yes_DECEASED']),'With Tx_dead')
print(len(result3.loc['MALE_No_DECEASED']),'Without Tx_dead')
print(len(result3.loc['MALE_Yes_LIVING']),'With Tx_dead')
print(len(result3.loc['MALE_No_LIVING']),'Without Tx_dead')'''

#print(len(result3.loc['FEMALE_Favorable_0']),'Female alive')
#print(len(result3.loc['FEMALE_Favorable_1']),'Female dead')
#print(len(result3.loc['FEMALE_Intermediate/Normal_0']),'Normal_alive')
#print(len(result3.loc['FEMALE_Intermediate/Normal_1']),'Normal_dead')

#Do PCA and calculate eigenvectors
pca = PCA(n_components=5)

transf=pca.fit_transform(result3)
var_exp=pca.explained_variance_ratio_
f = open('PCA.txt', 'w')
strings=''
eig_vecs=[]
for n in pca.components_:
    eig_vecs.append(n)
    strings+='\nComponent\n'
    for a in n:
        strings+=str(a)+'\t'
    

# calculate explain variance
stringv=''
print len(eig_vecs[0]),'Longueur de la matrice composante 1'
for n in pca.explained_variance_ratio_:
    stringv+=str(n)+'\t'
f.write('\n\nEigenvectors\n\n')
f.write(strings)
f.write('\n\nVariance\n\n')
f.write(stringv)
f.close()
cum_var_exp = np.cumsum(var_exp)

##########################################################################################################################################################
#Plot all groups on PCA (Don't forget to put plotly's access keys to share data over internet)
##########################################################################################################################################################

traces = []
name=['FEMALE_No_DECEASED','FEMALE_No_LIVING','FEMALE_Yes_DECEASED', 'FEMALE_Yes_LIVING','MALE_No_DECEASED','MALE_No_LIVING','MALE_Yes_DECEASED','MALE_Yes_LIVING']
for n in range (0,4):
    if n==0:
        trace = Scatter(
                        x=transf[0:41,n],
                        y=transf[0:41,n+1],
                        mode='markers',
                        name=name[n],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[41:61,n],
                        y=transf[41:61,n+1],
                        mode='markers',
                        name=name[n+1],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                            opacity=0.8))
        traces.append(trace)
        continue
    if n==1:
        trace = Scatter(
                        x=transf[61:79,n-1],
                        y=transf[61:79,n],
                        mode='markers',
                        name=name[n+1],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[79:84,n-1],
                        y=transf[79:84,n],
                        mode='markers',
                        name=name[n+2],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        continue
    if n==2:
        trace = Scatter(
                        x=transf[84:135,n-2],
                        y=transf[84:135,n-1],
                        mode='markers',
                        name=name[n+2],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[135:157,n-2],
                        y=transf[135:157,n-1],
                        mode='markers',
                        name=name[n+3],
                        marker=Marker(
                                        size=12,
                                        line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                        opacity=0.8))
        traces.append(trace)
        continue
    else:
        trace = Scatter(
                        x=transf[157:171,n-3],
                        y=transf[157:171,n-2],
                        mode='markers',
                        name=name[n+3],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[171:179,n-3],
                        y=transf[171:179,n-2],
                        mode='markers',
                        name=name[n+4],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                            yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
#plot_url = py.plot_mpl(fig)
py.plot(fig,filename='PCA1_Sex_TX_state.png')

trace1 = Bar(
             x=['PC %s' %i for i in range(1,6)],
             y=var_exp,
             showlegend=False)

trace2 = Scatter(
                 x=['PC %s' %i for i in range(1,6)],
                 y=cum_var_exp,
                 name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
              yaxis=YAxis(title='Explained variance in percent'),
              title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.plot(fig)


#plt.plot(transf[0:91,0], transf[0:91,1],
#         'o', markersize=7, color='blue', alpha=0.5, label='PC1')
#plt.plot(transf[91:179,0], transf[91:179,1],
#         '^', markersize=7, color='red', alpha=0.5, label='PC2')
#plt.xlim([-100,100])
#plt.ylim([-100,100])
#plt.xlabel('x_values')
#plt.ylabel('y_values')
#plt.legend()
#plt.title('Transformed samples with class labels')

#plt.show()
##########################################################################################################################################################
# Plot wih and without treatment groups
##########################################################################################################################################################

traces = []
name=['Without Tx','With Tx']
for n in range (0,2):
    if n==0:
        trace = Scatter(
                    x=transf[0:151,n],
                    y=transf[0:151,n+1],
                    mode='markers',
                    name=name[n],
                    marker=Marker(
                                  size=12,
                                  line=Line(
                                            color='rgba(217, 217, 217, 0.14)',
                                            width=0.5),
                                  opacity=0.8))
    else:
        trace = Scatter(
                        x=transf[151:200,n-1],
                        y=transf[151:200,n],
                        mode='markers',
                        name=name[n],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                            yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
#plot_url = py.plot_mpl(fig)
py.plot(fig,filename='PCA1_t.png')

##########################################################################################################################################################
#plot groups with treatments with survival state
##########################################################################################################################################################

traces = []
name=['NO_DECEASED','NO_LIVING','YES_DECEASED','YES_LIVING']
for n in range (0,2):
    if n==0:
        trace = Scatter(
                        x=transf[0:35,n],
                        y=transf[0:35,n+1],
                        mode='markers',
                        name=name[n],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[35:133,n],
                        y=transf[35:133,n+1],
                        mode='markers',
                        name=name[n+1],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
    else:
        trace = Scatter(
                        x=transf[133:147,n-1],
                        y=transf[133:147,n],
                        mode='markers',
                        name=name[n+1],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
        traces.append(trace)
        trace = Scatter(
                        x=transf[147:200,n-1],
                        y=transf[147:200,n],
                        mode='markers',
                        name=name[n+2],
                        marker=Marker(
                                      size=12,
                                      line=Line(
                                                color='rgba(217, 217, 217, 0.14)',
                                                width=0.5),
                                      opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                            yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
#plot_url = py.plot_mpl(fig)
py.plot(fig,filename='PCA1_Tx_State.png')


#print pca.components_,'Eigenvectors'
#print pca.explained_variance_ratio_, 'variance'

#result = pd.merge(DF1,data_gen , on='sampleID', how='outer')

#DF1= DF1.rename(columns = {'sampleID': 'sample'})
#result = pd.concat([data_gen,DF1], axis=1)
#print result
#transposed_data_gen['sample']=transposed_data_gen[0]
#print list(transposed_data_gen.columns.values)
#transposed_data_gen=transposed_data_gen.rename(columns = {'0': 'sampleID'})
#print transposed_data_gen[0:2]

