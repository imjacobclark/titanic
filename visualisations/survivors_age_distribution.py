import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/titanic_train.csv')

survived_passengers = dataset['Survived'] == 1

survived_passengers_ages = dataset[survived_passengers]['Age']

survived_passengers_ages_with_missing_filled = survived_passengers_ages.fillna(survived_passengers_ages.median())

age_counts = survived_passengers_ages_with_missing_filled.value_counts().sort_index()

labels = age_counts.index.values

plt.bar(np.arange(len(age_counts)), age_counts, align='center')
plt.xticks(np.arange(len(age_counts)), labels)
plt.title("Titanic Survivors: Men compared to Women")

plt.show()
