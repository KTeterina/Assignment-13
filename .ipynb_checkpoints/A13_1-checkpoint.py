# 1.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

income = np.array([23400, 20000, 19600, 16700, 20000, 15000, 14000, 13000, 13500, 16000])
income_without_tax = income * 0.7
expenses = np.array([8000, 9000, 11000, 10000, 8000, 12000, 13000, 9000, 14000, 12000])
months = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])

data = {"Income": income, "Income_without_tax": income_without_tax, "Expenses": expenses}

df = pd.DataFrame(data, index = months)

df["Savings"] = df["Income_without_tax"] - df["Expenses"]
df["Profit"] = df["Savings"] > 0

print(df)

# icnome
plt.figure(figsize=(7, 3)) # Adjust figure size
plt.bar(df.index, df["Income"], color=['#1344c6'])
plt.xlabel("Month")
plt.ylabel("Income")
plt.title("Income by Month")
plt.show()

#savings
positive_indices = df[df["Savings"] > 0].index
negative_indices = df[df["Savings"] < 0].index

plt.figure(figsize=(7, 3))
plt.bar(positive_indices, df["Savings"][positive_indices], color='#27af0c') #green
plt.bar(negative_indices, df["Savings"][negative_indices], color='#c62b13') #red
plt.xlabel("Month")
plt.ylabel("Savings")
plt.title("Savings by Month")
plt.show()

#pie chart
total_savings = df["Savings"].sum()
savings_percentages = (df["Savings"] / total_savings) * 100
#print(savings_percentages) # jun, jul, sep, oct

def multiply_negative_values(savings_percentages):
    modified_percentages = []
    for percentage in savings_percentages:
        if percentage < 0:
            modified_percentages.append(percentage * -1)
        else:
            modified_percentages.append(percentage)
    return modified_percentages

modified_percentages = multiply_negative_values(savings_percentages)
print(f"Original percentages: \n{savings_percentages}")
print(f"Modified percentages: \n{modified_percentages}")

import matplotlib.cm as cm
cmap = cm.get_cmap('Blues') # Example using 'tab10'

colors = cmap(np.linspace(0, 1, len(df.index)))

plt.figure(figsize=(4, 4))
plt.pie(modified_percentages, labels=df.index, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title("Monthly Savings Proportion in Total Savings")
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Output avg income by quarters
import statistics as st
income = df["Income"]

print("1st Quarter avarage income:", st.mean((income.iloc[:3])))

print("\n2st Quarter avarage income:", st.mean((income.iloc[3:6])))

print("\n3st Quarter avarage income:", st.mean((income.iloc[6:9])))

print("\n4st Quarter avarage income:", st.mean((income.iloc[9:])))

#1.2
import seaborn as sns
car_crashes = sns.load_dataset('car_crashes')
print(car_crashes.head())

#plot 1
plt.figure(figsize=(7, 3))
plt.hist(car_crashes["ins_losses"],  bins=10, color='#1344c6', edgecolor='black')
plt.xlabel("Losses (per insured driver)")
plt.ylabel("Frequency")
plt.title("Histogram of incurance companies losses")
plt.show()

#plot 2
plt.figure(figsize=(7, 3))
plt.scatter(car_crashes["speeding"], car_crashes["alcohol"], c = car_crashes["ins_premium"], s=80, label = 'Car insurance premiums')
plt.legend(loc='upper left', fontsize=12, frameon=True)
plt.xlabel("Percentage of fatal crashes with speeding")
plt.ylabel("Alcohol impaired fatal crashes")
plt.title("Scatter plot of fatal crashes with speeding, alcohol and car insurance premiums")
plt.show()

plt.figure(figsize=(7, 3))
sns.scatterplot(data = car_crashes, x = 'speeding', y = 'alcohol', hue = 'ins_premium', s = 80, label = 'Car insurance premiums')
plt.legend()
plt.xlabel("Percentage of fatal crashes with speeding")
plt.ylabel("Alcohol impaired fatal crashes")
plt.title("Scatter plot of fatal crashes with speeding, alcohol and car insurance premiums")
plt.show()

#plot3
plt.figure(figsize=(10, 5))
plt.bar(car_crashes["abbrev"], car_crashes["not_distracted"])
plt.xticks(rotation=90)
plt.xlabel("American states")
plt.ylabel("Percentage of non-distracted drivers")
plt.title("Histogram of fatal crushes of non-distracted drivers by states")
plt.show()