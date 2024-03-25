from mpl_toolkits.mplot3d import Axes3D from  sklearn.preprocessing 
import StandardScaler    
import matplotlib.pyplot as plt # plotting import numpy as np # linear  
algebra import os # accessing directory structure import pandas as pd # 
data processing, CSV file I/O (e.g. pd.read_csv) for dirname, _, filenames 
in os.walk('/kaggle/input'):     for filename in filenames:    
print(os.path.join(dirname, filename))    
# 
Distribution graphs (histogram/bar graph) of column data def plotPerColumnDistribution(df, 
nGraphShown, nGraphPerRow):       
nunique = df.nunique()    
df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick  
columns that have between 1 and 50 unique values     nRow, nCol = df.shape     columnNames = list(df)      
nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow    
plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w',  
edgecolor = 'k')
     for i in range(min(nCol, nGraphShown):    
plt.subplot(nGraphRow, nGraphPerRow, i + 1)       
  columnDf 
= df.iloc[:, i]         
np.number)):               
if (not np.issubdtype(type(columnDf.iloc[0]), 
valueCounts = columnDf.value_counts()               
valueCounts.plot.bar()    
     else:    
columnDf.hist()         
plt.xticks(rotation = 90)  
plt.ylabel('counts')          
       plt.title(f'{columnNames[i]} 
(column {i})')     
h_pad = 2)     
plt.tight_layout(pad = 2, w_pad = 2, 
plt.show() # Correlation matrix def 
plotCorrelationMatrix(df, graphWidth):     
df.dataframeName     
filename = 
df = df.dropna('columns') # drop 
columns with NaN    
df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique 
values     if df.shape[1] < 2:    
return     
print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is  less 
than 2')         
corr = df.corr()     
facecolor='w', 
edgecolor='k')     
plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, 
corrMat 
= plt.matshow(corr, 
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)     
fignum 
= 1)     
plt.yticks(range(len(corr.columns)), 
corr.columns)     
plt.gca().xaxis.tick_bottom()     
{filename}', fontsize=15)
     plt.show()    
plt.colorbar(corrMat)     
# Scatter and density plots def plotScatterMatrix(df, plotSize, textSize):     
df.select_dtypes(include =[np.number]) # keep only numerical columns      
plt.title(f'Correlation Matrix for 
df = 
# Remove 
rows and columns that would lead to df being singular
     df = df.dropna('columns')    
df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique  
values    
columnNames = list(df)     if len(columnNames) > 10: # reduce the number of columns for matrix  
inversion of kernel density plots         
columnNames = columnNames[:10]     df = df[columnNames]     ax = 
corrs = 
pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')     
df.corr().values     
for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):    
ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',  
va='center', size=textSize)    
plt.suptitle('Scatter and Density Plot')
     plt.show()  nRowsRead 
= 1000 # specify 'None' if want to read whole file    
# employee_reviews.csv may have more rows in reality, but we are only loading/previewing the first 1000 
rows    
df1 
= pd.read_csv('/content/employee_reviews.csv.zip', 
delimiter=',', 
nrows 
df1.dataframeName = '/content/employee_reviews.csv.zip' nRow, nCol = df1.shape   
{nRow} rows and {nCol} columns') df1.head(5)  
plotPerColumnDistribution(df1, 10, 5)