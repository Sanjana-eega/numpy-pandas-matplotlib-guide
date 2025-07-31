import pandas as pd
df=pd.read_csv("/content simple_used_car_dataset.csv")
df
//Target value/which is going to predict is the dependent variable(price).remaining all the variables are independent.

df.shape
//give rows and column count

Df.head()
//return top 5 values

Df.tail()
//return down 5 values

Df.describe()
//return all the static methods.like count, max,25%..

import pandas as pd
df=pd.read_csv("/content/simple_used_car_dataset.csv")
df[“Price”]
//return the column of price

df[['Price','Year']]
//to read 2 column

df[‘year’].mean()
//calculate entire column mean

df['Fuel_Type'].fillna(df['Fuel_Type'].mode()[0],inplace=True)
//Fuel type has the repeated values. So we take mode here to fill the null values.for kilometres we have taken mean.

df[df.duplicated()]
//to check the duplicate values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Fuel_Type']=le.fit_transform(df[['Fuel_Type']])
df
//We are changing the text to the number by using scaling in the Fuel_Type column like diesel is 1 and petrol is 0

------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
*Data cleaning(7) in data science involves:-
=>missing value detect and treat
=>outliers
=>identify duplicate =>feature encoding
=>feature scaling
=>feature selection
=>feature extraction
The raw data will not be decoded by the ai. So we are using data cleaning.
(AI+X=project) in which x is the domain.  Applying ml or ai to domain then project (client have)is done. first we feed the data which is divide into primary (its our data ) and secondary(taken from somewhere else)
Take dataset (create dummy) and apply model


*DATA VISUALIZATION
Pick the libraries/tools in data visualisation.
Sklearn handle the entire Eda process and algorithm. plotly made data 3d.


*Model building=> model depend on data.
 If We have css, excel use ml. If we have a text or audio, go for the nlp. If we have image or video then go for cnn. 

In supervised:

regression(we have matrices evaluation we use rmse,mse,mae,r^2)=>linear regressison,decision tree, random forest regression,knnr,svmr

classification=> logistic regression,rfc(accuracy, precision,f1score,recall),dtc,knnc,svm classifier

When we are applying nlp algorithm this matters: (latm,rrna,rna)

When we applying can: can algorithms is required

*deployement+monitor

*ML ops

-----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------

*numpy=> it’s a manipulator (control)which manipulate the array
In collab, we have by default gpu. Where the system won’t crash. In setting, miscelleneous,on all the thing and put many power(magic🪄)
Import numpy as np
a=np.array([5,4,8,7]) 
//creating array and calling

A.shape
//return length

print(type(a))
//return datatype

a=np.array([[5,4,8,7],[2,3,9,0]])
//creating 2d array

a[0,1]
//fetching in 2d array

a[:,::-1]
//for reverse

a.T
//in built function used to make pairs

np.std(a)
//standard deviation like this only we have sum,min, max,abs(absolute value),var

np.ones((3,3))
//inside a 3*3 matrix which is filled with ones

np.full((3,3),6)
//array([[6, 6, 6],
       [6, 6, 6],
       [6, 6, 6]]

import pandas as pd
import numpy as np
f={
    'd':[1,2,3]
}
f
//{'d': [1, 2, 3]}


//we converted the given data in the form of dictionary into rows and columns

import pandas as pd
import numpy as np
f={
    'd':[1,2,3],
    'a':[4,5,6]
}
df=pd.DataFrame(f)
df
//
	d	a
0	1	4
1	2	5
2	3	6

Np.nan
//to give the null values in 'd':[1,2,3,np.nan]. Nan is missing value here

df.isnull().sum()
//identifies/finding  a missing value

df[“d”]
//return the selected column

df[“d”].fillna(7,inplace=True)
Df
//replacing 7 with null

df[“d”].fillna(,inplace=True)
//by-default it fill

df[“a”].fillna(df['a'],mean,inplace=True)
df
//fill with mean value

df[“k”].fillna(df[‘k’],mode()[0],inplace=True)


//treat the missing value (replace it with something(10 or 20% data is missing) by using mean,mode,median or just drop by dropna(used when the 90% data is missing))


PANDAS IS DATA MANIPULATOR

df.info()
//getting info of data

df.describe()

Import pandas as pd
Import numpy as np
f={
     ‘Age’:[22,33,44],
      ‘Level’:[‘swab’,’bdbd’,’swab’]
}
df=pd.DataFrame(f)
df
//find a duplicate values and here we are taken a dummy data and which is converting into rows and columns 

df.duplicated()
//return true false

df[df.duplicated()]
//return duplicate value

Df.drop_duplicates(subset=‘age’,inplace=True)
Df
//delete the duplicate(treatment of duplicates)

=>FETURE ENCODER WHICH ENCODE CATEGORICAL DATA INTO NUMBERS

From sklearn.preprocessing import LabelEncoder()
le=LabelEncoder()
df[‘level’]=le.fit_transform(d[[‘level’]])
Df
//we stored the fn labelencounter in le. And give the ranking acc to data

From sklearn.preprocessing import LabelEncoder(),OneHotEncoder
le=OneHotEncoder
df[‘level’]=le.fit_transform(df[[‘level’]]).toarray()
df
//it puts zeroes and ones

=>FEATURES SCALE 

Use minmaxscaler (0 to 1)and standard scaler(standard deviation is 1 to mean is 0)

From sklearn.preprocessing import MinMaxScaler,StandardScaler
mm=MinMaxScaler()
df[‘age’]=mm.fit_transform(df[[‘age’]])
df.describe() or df
