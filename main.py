import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from derive_rules_pure import RIPPER, ID3, CN2, PART, split_columns, cv_accuracy

os.makedirs("output", exist_ok=True)


# sintetiskai sugeneruotas duomenu rinkinis
df = pd.read_csv("data/loans.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nClasses:")
print(df["decision"].value_counts())
print("\nStatistics:")
print(df.describe())


target = "decision"
num_cols, cat_cols = split_columns(df, target)

# split 70% train / 30% test
test = df.groupby(target).sample(frac=0.3, random_state=42)
train = df.drop(test.index).reset_index(drop=True)
test = test.reset_index(drop=True)


methods = {"JRip": RIPPER, "ID3": ID3, "CN2": CN2, "PART": PART}

results = []
for name, method in methods.items():
    print("\n==========", name, "==========")
    model = method().fit(train, target, num_cols, cat_cols)

    rules = model.rules_text()
    print(rules)
    with open("output/rules_" + name + ".txt", "w") as f:
        f.write(rules + "\n")

    train_acc = np.mean(model.predict(train) == train[target].to_numpy())
    test_acc = np.mean(model.predict(test) == test[target].to_numpy())

    cv_acc = cv_accuracy(method, df, target, num_cols, cat_cols, 10, 1) / 100

    lines = rules.split("\n")
    lengths = [line.count(" AND ") + 1 if line.startswith("IF") else 0 for line in lines]

    results.append({
        "method": name,
        "train accuracy": round(train_acc, 3),
        "test accuracy": round(test_acc, 3),
        "10-fold CV accuracy": round(cv_acc, 3),
        "number of rules": len(lines),
        "avg rule length": round(np.mean(lengths), 2),
    })

results = pd.DataFrame(results)
print("\n========== Comparison ==========")
print(results.to_string())
results.to_csv("output/comparison.csv", index=False)


plt.figure(figsize=(7, 5))
plt.scatter(results["number of rules"], results["test accuracy"])
for i in range(len(results)):
    plt.annotate(results["method"][i], (results["number of rules"][i], results["test accuracy"][i]))
plt.xlabel("Number of rules")
plt.ylabel("Test accuracy")
plt.title("Accuracy vs number of rules")
plt.grid()
plt.savefig("output/accuracy_vs_rules.png")
plt.close()
