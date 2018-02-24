import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model



report = pd.read_csv('train.tsv', sep='\t', names = ['price','isNew','rooms','floor','location','sqrMetres'])

print(report.describe())

print(report.columns)

#report.plot()
#plt.show()

#sns.regplot(y=report['price'],x=report['sqrMetres'])
#plt.show()

reg = linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['sqrMetres']), report['price'])

Predict=reg.predict(50)

print (Predict)