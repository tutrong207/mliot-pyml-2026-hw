import seaborn as sns
import pandas as pd

df = sns.load_dataset("iris")

#print(df.head())
#print(df.shape)
#print(df.dtypes)

#Phần 1
#5 dòng đầu

print(df.iloc[[0,1,2,3,4]])

print("Mean 1:", df['sepal_length'].mean())
print("Median 1:", df['sepal_length'].median())
print("Mode 1:", df['sepal_length'].mode()[0])
print("Variance 1:", df['sepal_length'].var())
print("Standard deviation 1:", df['sepal_length'].std())
print("Min 1:", df['sepal_length'].min())
print("Max 1:", df['sepal_length'].max())
print("Q1 1:", df['sepal_length'].quantile(0.25))
print("Q3 1:", df['sepal_length'].quantile(0.75))
Q1_1 = df['sepal_length'].quantile(0.25)
Q3_1 = df['sepal_length'].quantile(0.75)
print("IQR 1:", Q3_1-Q1_1)


print("Mean 2:", df['sepal_width'].mean())
print("Median 2:", df['sepal_width'].median())
print("Mode 2:", df['sepal_width'].mode()[0])
print("Variance 2:", df['sepal_width'].var())
print("Standard deviation 2:", df['sepal_width'].std())
print("Min 2:", df['sepal_width'].min())
print("Max 2:", df['sepal_width'].max())
print("Q1 2:", df['sepal_width'].quantile(0.25))
print("Q3 2:", df['sepal_width'].quantile(0.75))
Q1_2 = df['sepal_width'].quantile(0.25)
Q3_2 = df['sepal_width'].quantile(0.75)
print("IQR 2:", Q3_2-Q1_2)


print("Mean 3:", df['petal_length'].mean())
print("Median 3:", df['petal_length'].median())
print("Mode 3:", df['petal_length'].mode()[0])
print("Variance 3:", df['petal_length'].var())
print("Standard deviation 3:", df['petal_length'].std())
print("Min 3:", df['petal_length'].min())
print("Max 3:", df['petal_length'].max())
print("Q1 3:", df['petal_length'].quantile(0.25))
print("Q3 3:", df['petal_length'].quantile(0.75))
Q1_3 = df['petal_length'].quantile(0.25)
Q3_3 = df['petal_length'].quantile(0.75)
print("IQR 3:", Q3_3-Q1_3)


print("Mean 4:", df['petal_width'].mean())
print("Median 4:", df['petal_width'].median())
print("Mode 4:", df['petal_width'].mode()[0])
print("Variance 4:", df['petal_width'].var())
print("Standard deviation 4:", df['petal_width'].std())
print("Min 4:", df['petal_width'].min())
print("Max 4:", df['petal_width'].max())
print("Q1 4:", df['petal_width'].quantile(0.25))
print("Q3 4:", df['petal_width'].quantile(0.75))
Q1_4 = df['petal_width'].quantile(0.25)
Q3_4 = df['petal_width'].quantile(0.75)
print("IQR 4:", Q3_4-Q1_4)


print("Setosa:", df[df['species'] == 'setosa']['sepal_length'].agg(['mean', 'std']))
