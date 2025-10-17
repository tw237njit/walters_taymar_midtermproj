# ----------------------------------------------------------
# Market Basket Analysis: Brute Force, Apriori, FP-Growth
# ----------------------------------------------------------
# Author: Taymar Walters
# Description:
#   1. Load and parse a transactions CSV file.
#   2. Run a brute-force frequent itemset mining algorithm.
#   3. Generate association rules from brute-force results.
#   4. Run Apriori and FP-Growth using mlxtend for comparison.
# ----------------------------------------------------------

import itertools
import subprocess
import sys

# ==========================================================
# STEP 0: Auto-install required packages (if missing)
# ==========================================================
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["pandas", "mlxtend"]:
    install_if_missing(pkg)

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
# ==========================================================
# STEP 0: Load Dataset (with file path check)
# ==========================================================




# ==========================================================
# STEP 1: Load Dataset (with file path check)
# ==========================================================
import os
import pandas as pd
print

try:
    chosenDatabase = int(input("Enter number to select a database"))
except ValueError:
    print("Invalid Input Please try again")
 
# Always load relative to script location
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "generic_transactions.csv")
data = pd.read_csv(file_path)

#filename = r"C:\SCHOOL PROJECTS\Data Mining\walters_taymar_midtermproj\generic_transactions.csv"

try:
    data = pd.read_csv(file_path)
    print("✅ File loaded successfully!\n")
except FileNotFoundError:
    raise FileNotFoundError("❌ File not found. Please check your file path and try again.")

# ==========================================================
# STEP 2: Parse transactions
# ==========================================================
transactions = [t.replace(" ", "").split(",") for t in data["Transaction"]]
all_items = sorted(set(item for sublist in transactions for item in sublist))

# =====================================================
# USER INPUT SECTION
# =====================================================

try:
    min_support = float(input("Enter minimum support (e.g., 0.3 for 30%): "))
except ValueError:
    print("Invalid input. Using default min_support = 0.3")
    min_support = 0.3

try:
    min_confidence = float(input("Enter minimum confidence (e.g., 0.6 for 60%): "))
except ValueError:
    print("Invalid input. Using default min_confidence = 0.6")
    min_confidence = 0.6

print(f"\nUsing min_support = {min_support} and min_confidence = {min_confidence}\n")

# ==========================================================
# STEP 3: Brute-force Frequent Itemset Mining
# ==========================================================
def get_support(itemset, transactions):
    """Compute support count for a given itemset."""
    return sum(1 for t in transactions if set(itemset).issubset(set(t)))

def brute_force_mining(transactions, min_support):
    num_transactions = len(transactions)
    frequent_itemsets = []
    k = 1

    while True:
        candidates = [list(i) for i in itertools.combinations(all_items, k)]
        level_frequent = []
        for c in candidates:
            support = get_support(c, transactions) / num_transactions
            if support >= min_support:
                level_frequent.append((tuple(c), support))

        if not level_frequent:
            break

        frequent_itemsets.extend(level_frequent)
        print(f"Found {len(level_frequent)} frequent {k}-itemsets")
        k += 1

    return frequent_itemsets

# Run brute-force mining
min_support = 0.3
frequent_itemsets_brute = brute_force_mining(transactions, min_support)



# ==========================================================
# STEP 4: Generate Association Rules (Brute Force)
# ==========================================================
def generate_rules(frequent_itemsets, min_confidence, transactions):
    num_transactions = len(transactions)
    rules = []

    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(set(itemset) - set(antecedent))
                sup_itemset = get_support(itemset, transactions) / num_transactions
                sup_antecedent = get_support(antecedent, transactions) / num_transactions
                confidence = sup_itemset / sup_antecedent if sup_antecedent > 0 else 0

                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, sup_itemset, confidence))
    return rules

min_confidence = 0.6
rules_brute = generate_rules(frequent_itemsets_brute, min_confidence, transactions)

# ==========================================================
# STEP 5: Convert to One-Hot Encoding for MLXTEND
# ==========================================================
one_hot = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
for i, t in enumerate(transactions):
    one_hot.loc[i, t] = 1

# ==========================================================
# STEP 6: Run Apriori and FP-Growth
# ==========================================================
frequent_itemsets_ap = apriori(one_hot, min_support=min_support, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=min_confidence)

frequent_itemsets_fp = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)

# ==========================================================
# STEP 7: Display Association Rules (All Algorithms)
# ==========================================================
print("\n\n================================================")
print("🔹 FREQUENT ITEMS FOUND BY BRUTE FORCE:")
print("================================================")
for items, sup in frequent_itemsets_brute:
    print(f"{items} | support: {sup:.2f}")
print("\n\n================================================")
print("🔸 ASSOCIATION RULES — BRUTE FORCE")
print("================================================")
for ant, cons, sup, conf in rules_brute:
    print(f"{ant} → {cons} (support: {sup:.2f}, confidence: {conf:.2f})")

print("\n\n================================================")
print("🔸 ASSOCIATION RULES — APRIORI")
print("================================================")
for _, row in rules_ap.iterrows():
    print(f"{tuple(row['antecedents'])} → {tuple(row['consequents'])} "
          f"(support: {row['support']:.2f}, confidence: {row['confidence']:.2f})")

print("\n\n================================================")
print("🔸 ASSOCIATION RULES — FP-GROWTH")
print("================================================")
for _, row in rules_fp.iterrows():
    print(f"{tuple(row['antecedents'])} → {tuple(row['consequents'])} "
          f"(support: {row['support']:.2f}, confidence: {row['confidence']:.2f})")

print("\n✅ All algorithms executed successfully!")
