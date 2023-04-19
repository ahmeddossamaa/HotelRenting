from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sns as sns
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm






"""
def anova(x, y, k):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(x, y)
    return selector.get_support(indices=True)"""
def categToNumerical(data , l):
    p_value=[]
    actual=[]
    new=[]
    names = ['Hotel_Name', 'Reviewer_Nationality', 'room_type', 'trv_type', 'rev_month', 'rev_day']

    for i in names:
        new.append(stats.f_oneway(data[i],l,axis=0))
        p_value.append(stats.ttest_ind(data[i], data.iloc[:, 9]))

    #for k in range (9):
     #   if p_value[k].pvalue<0.5:
      #      actual.append(p_value[k])

  #  plt.subplots(figsize=(len(data.columns), len(data.columns)))
   # sns.heatmap(actual, annot=True)
    #plt.show()
    return p_value, new

def pearson(d,x ,t, l):
    corr = d.corr(x[t],method= 'pearson', min_periods= 1)

    return corr

"""
def NumericalCorrelation(data):

    corr = data.corr(method='spearman')
    print("corr", corr)
    corr.info()
    top_feature = corr.loc[abs(corr['Reviewer_Score']) > 0.5]
    print("top_feature", top_feature)
    # Correlation plot
    plt.subplots(figsize=(len(data.columns), len(data.columns)))
    # top_corr = scaled_df[top_feature].corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    return top_feature

"""

data=pd.read_csv("C:/Users/SCH/Desktop/data/hotel-regression-dataset (2).csv")
x = pd.read_csv("C:/Users/SCH/Desktop/data/test-preprocessing-v3.csv")
x = x[['Hotel_Name', 'Reviewer_Nationality', 'room_type', 'trv_type', 'rev_month', 'rev_day', 'days_number', 'trip_type', 'rev_year']]
    #x['trip_type'], x['rev_year']
x['review']= data['Reviewer_Score']
print( x)

sper=[]
new=[]
l= data['Reviewer_Score']
#ano = anova(x.iloc[:,7:9],data['Reviewer_Score'],1)
ano,new = categToNumerical(x,l)
sper.append(pearson(x.iloc[:,7],x,'review' , 0))
sper.append(pearson(x.iloc[:,8],x,'review' , 0))
print (ano)
print (new)
print (sper)
