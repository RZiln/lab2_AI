# Script that generates the synthetic loan dataset (run once, result is saved
# to data/loans.csv and then used in main.py)

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data", exist_ok=True)

n = 1000

age = np.random.randint(18, 71, n)
income = np.random.lognormal(np.log(2500), 0.5, n).round(-1)  # monthly income in EUR
income = np.clip(income, 600, 15000)
credit_score = np.random.normal(650, 100, n).round()
credit_score = np.clip(credit_score, 300, 850)
debt_ratio = np.random.beta(2, 3, n).round(3)  # part of income that goes to debts
loan_amount = np.random.uniform(1000, 60000, n).round(-2)
has_collateral = np.random.choice(["yes", "no"], n, p=[0.4, 0.6])
purpose = np.random.choice(["car", "house", "education", "business"], n)
employment_years = np.random.exponential(6, n).round(1)
employment_years = np.minimum(employment_years, age - 18)  # can't work longer than since 18

# Rules that I used to make the class (the "true" rules):
# 1) credit_score < 550 -> rejected
# 2) debt_ratio > 0.5 and no collateral -> rejected
# 3) loan_amount > 12 * income and employment_years < 2 -> rejected
# 4) everything else -> approved
decision = []
for i in range(n):
    if credit_score[i] < 550:
        decision.append("rejected")
    elif debt_ratio[i] > 0.5 and has_collateral[i] == "no":
        decision.append("rejected")
    elif loan_amount[i] > 12 * income[i] and employment_years[i] < 2:
        decision.append("rejected")
    else:
        decision.append("approved")

# add some noise (5% of labels are flipped) so it's more like real data
for i in range(n):
    if np.random.rand() < 0.05:
        if decision[i] == "approved":
            decision[i] = "rejected"
        else:
            decision[i] = "approved"

df = pd.DataFrame({
    "age": age,
    "income": income,
    "employment_years": employment_years,
    "credit_score": credit_score,
    "debt_ratio": debt_ratio,
    "loan_amount": loan_amount,
    "has_collateral": has_collateral,
    "purpose": purpose,
    "decision": decision,
})

# make some values missing (about 2%) to have something to clean
for col in ["income", "employment_years", "has_collateral"]:
    missing = np.random.rand(n) < 0.02
    df.loc[missing, col] = np.nan

df.to_csv("data/loans.csv", index=False)
print("Saved", len(df), "rows to data/loans.csv")
