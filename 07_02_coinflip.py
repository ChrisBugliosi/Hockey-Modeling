import random
from pandas import DataFrame, Series
import statsmodels.formula.api as smf


coin = ['H', 'T']

# make empty DataFrame
df = DataFrame(index=range(100))

# now fill it with a "guess"
df['guess'] = [random.choice(coin) for _ in range(100)]

# and flip
df['result'] = [random.choice(coin) for _ in range(100)]

# did we get it right or not?
df['right'] = (df['guess'] == df['result']).astype(int)

# regression
model = smf.ols(formula='right ~ C(guess)', data=df)
results = model.fit()
results.summary2()

random.randint(1, 10)

def run_sim_get_pvalue(n):
    coin = ['H', 'T']
    df = DataFrame(index=range(n))
    df['guess'] = [random.choice(coin) for _ in range(n)]
    df['result'] = [random.choice(coin) for _ in range(n)]
    df['right'] = (df['guess'] == df['result']).astype(int)
    model = smf.ols(formula='right ~ C(guess)', data=df)
    results = model.fit()
    return results.pvalues['C(guess)[T.T]']

print(run_sim_get_pvalue(100))

result_1k = Series(run_sim_get_pvalue(100) for _ in range(1000))
print(result_1k.mean())

def run_till_threshold(i, p=0.05):
    pvalue = run_sim_get_pvalue(100)
    if pvalue < p:
        return i
    else:
        return run_till_threshold(i+1, p)

results = Series([run_till_threshold(1) for _ in range(100)])
print(results.mean())
print(results.median())