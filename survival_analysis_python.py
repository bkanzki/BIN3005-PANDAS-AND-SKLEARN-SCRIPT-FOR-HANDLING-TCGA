import pandas as pd
## need to install it  https://github.com/CamDavidsonPilon/lifelines
from lifelines.estimation import KaplanMeierFitter  
from lifelines import CoxPHFitter
from lifelines.statistics import *
from lifelines.utils import k_fold_cross_validation
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

df = pd.read_table('clinical_data.tab',sep='\t')
#df2= pd.read_table('genomicMatrix.tab',sep='\t')
#print list(df.columns.values)

    
survival_col = '_OS'
censor_col = '_OS_IND'
clinical_predictors = ['age_at_initial_pathologic_diagnosis']
df = df[pd.notnull(df[survival_col])]


tx = df['history_of_neoadjuvant_treatment']=='Yes'
ax = plt.subplot(111)

kmf1 = KaplanMeierFitter(alpha=0.95)
kmf1.fit(durations=df.ix[tx, survival_col], event_observed=df.ix[tx, censor_col], label=['Tx==Yes'])
kmf1.plot(ax=ax, show_censors=True,  ci_show=False)


kmf2 = KaplanMeierFitter(alpha=0.95)
kmf2.fit(durations=df.ix[~tx, survival_col], event_observed=df.ix[~tx, censor_col], label=['Tx==No'])
kmf2.plot(ax=ax, show_censors=True,  ci_show=False )

add_at_risk_counts(kmf1, kmf2, ax=ax)
plt.title ('Acute myeloid leukemia survival analysis with Tx and without Tx')
plt.xlabel(survival_col)
plt.savefig('km.png')

results = logrank_test(df.ix[tx, survival_col], df.ix[~tx, survival_col], df.ix[tx, censor_col], df.ix[~tx, censor_col], alpha=.99 )
results.print_summary()

cox = CoxPHFitter(normalize=False)
df_age = df[[survival_col, censor_col, 'age_at_initial_pathologic_diagnosis']]
df_age = df_age[pd.notnull(df_age['age_at_initial_pathologic_diagnosis'])]
cox = cox.fit(df_age, survival_col, event_col=censor_col, include_likelihood=True)
cox.print_summary()

scores = k_fold_cross_validation(cox, df_age, survival_col, event_col=censor_col, k=10)
print scores
print 'Mean score', np.mean(scores)
print 'Std', np.std(scores)
 