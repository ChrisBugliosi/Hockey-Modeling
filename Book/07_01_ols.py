import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from os import path

DATA_DIR = '/Users/chrisbugs/Downloads/code-hockey-files-0.2.1 2/data'

###################
# linear regression
###################

# load
df = pd.read_csv(path.join(DATA_DIR, 'shots.csv'))

# process
df['dist_sq'] = df['dist']**2
df['goal'] = df['goal'].astype(int)

print(df[['goal', 'dist', 'dist_sq']].head(20))

model = smf.ols(formula='goal ~ dist + dist_sq', data=df)
results = model.fit()

print(results.summary2())

model2 = model = smf.ols(formula='goal ~ dist + dist_sq + C(period)', data=df)
results2 = model2.fit()

print(results2.summary2())

def prob_of_goal(dist):
    b0, b1, b2 = results.params
    return b0 + b1 * dist + b2 * (dist ** 2)


df['goal_hat'] = results.predict(df)
print(df[['dist','goal', 'goal_hat']].head())

# Calculate the predicted probabilities for a range of distances
distances = range(0, 101)  # Adjust the range as needed
probabilities = [prob_of_goal(dist) for dist in distances]

# Create a DataFrame for the plot
prob_df = pd.DataFrame({'Distance': distances, 'Probability': probabilities})

# Plot the probabilities using Seaborn
sns.lineplot(data=prob_df, x='Distance', y='Probability')
plt.title('Probability of Scoring a Goal by Distance')
plt.xlabel('Distance from Goal (ft)')
plt.ylabel('Predicted Probability')
plt.grid(True)
# plt.show()

df['goal_hat_alt'] = df[['dist']].apply(prob_of_goal)
print(df['goal_hat_alt'].head())