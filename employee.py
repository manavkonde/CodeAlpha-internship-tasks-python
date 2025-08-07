import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv("unemployment.csv")
#print(df1.head())
df2=pd.read_csv("unemployment2020.csv")
#print(df2.head())

columns1 = set(df1.columns)
columns2 = set(df2.columns)

# Compare
# if columns1 == columns2:
#     print("✅ Columns are the same.")
# else:
#     print("❌ Columns are different.")
#     print("Only in file1:", columns1 - columns2)
#     print("Only in file2:", columns2 - columns1)

df1.drop(["Area"],axis=1,inplace=True)
df2.drop(['longitude', 'latitude', 'Region.1'],axis=1,inplace=True)
df3=pd.concat([df1,df2],ignore_index=True)
df3.to_csv("unemploymentdata.csv",index=False)
df4=pd.read_csv("unemploymentdata.csv")


df5=df4.dropna()
print(df5.columns)

df5.columns = df5.columns.str.strip()
df5['Date'] = pd.to_datetime(df5['Date'])
sns.lineplot(data=df5,x='Date',y='Estimated Unemployment Rate (%)',ci=None)
plt.figure(figsize=(12,6)) 
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Estimated Unemployment Rate Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

#filtering data
covid_19=df5[(df5['Date'].dt.year >= 2019) & (df5['Date'].dt.year <=2022)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=covid_19, x='Date', y='Estimated Unemployment Rate (%)', color='red',ci=None)
plt.axvline(pd.to_datetime('2020-03-01'), color='black', linestyle='--', label='COVID Start')
plt.title(' Unemployment Trend During COVID-19 (2019–2022)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.tight_layout()
plt.show()

#samary
pre_covid = df5[df5['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
post_covid = df5[df5['Date'] >= '2020-03-01']['Estimated Unemployment Rate (%)'].mean()

print(f"📉 Average Pre-COVID Rate: {pre_covid:.2f}%")
print(f"📈 Average Post-COVID Rate: {post_covid:.2f}%")


#monthly and yearly chart
df5['month']=df5['Date'].dt.month
df5['Year']=df5['Date'].dt.year

plt.figure(figsize=(12,6))
sns.barplot(data=df5,x='month',y='Estimated Unemployment Rate (%)')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df5, x='Year', y='Estimated Unemployment Rate (%)', estimator='mean', ci=None, marker='o')
plt.title('📊 Average Annual Unemployment Rate')
plt.xlabel('Year')
plt.ylabel('Average Unemployment Rate (%)')
plt.show()
