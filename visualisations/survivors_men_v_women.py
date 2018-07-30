import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/titanic_train.csv')

male_passengers = dataset['Sex'] == 'male'
female_passengers = dataset['Sex'] == 'female'
survived_passengers = dataset['Survived'] == 1

survived_males = dataset[male_passengers][survived_passengers]
survived_females = dataset[female_passengers][survived_passengers]

number_of_survived_males = survived_males.shape[0]
number_of_survived_females = survived_females.shape[0]

male_vs_female = [number_of_survived_males, number_of_survived_females]

labels = ("Men", "Women")

plt.bar(np.arange(len(male_vs_female)), male_vs_female, align='center')
plt.xticks(np.arange(len(male_vs_female)), labels)
plt.title("Titanic Survivors: Men compared to Women")

plt.show()
