# ----------------------------------------------------------
# Market Basket Analysis: Brute Force, Apriori, FP-Growth
# ----------------------------------------------------------
# Author: Taymar Walters
# Description:
#   1. Load and parse a transactions CSV file.
#   2. Run a brute-force frequent itemset mining algorithm.
#   3. Generate association rules from brute-force results.
#   4. Run Apriori and FP-Growth using mlxtend for comparison.
#   5. Track execution times for all algorithms.
# ----------------------------------------------------------

import itertools
import subprocess
import sys
import os
import pandas as pd
import time  # <-- Added for timing

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

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ==========================================================
# STEP 1: Select and Load Datasets 
# ==========================================================
print("Here are the following transactional databases\n"
      " 1) Generic\n 2) Nike\n 3) Best Buy\n 4) Coffee Shop\n 5) K-mart\n ")

def selectfile():
    while True:
        try:
            fileNumber = int(input("Enter number to select a database: \n"))
            match fileNumber:
                case 1:
                    return "generic_transactions.csv", "generic_items.csv"
                case 2:
                    return "nike_product_transactions.csv", "nike_products.csv"
                case 3:
                    return "bestbuy_transactions.csv", "bestbuy_products.csv"
                case 4:
                    return "coffee_transactions.csv", "coffee_items.csv"
                case 5:
                    return "k-mart_transactions.csv", "k-mart_items.csv"
                case _:
                    print("Invalid input. Please try again.")
        except ValueError:
            print("Please enter a valid number between 1–5.")

transactions, items = selectfile()  
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, items)

print("=============================================================")
print("Here are the unique items corresponding to the transactions:")
print("=============================================================")
df = pd.read_csv(file_path).dropna(how='all')
df.columns = df.columns.str.strip()
df["Item #"] = df["Item #"].astype(int)
print(df.to_string(index=False))
print("================================================")

file_path = os.path.join(base_path, transactions)
possible_cols = ["transaction", "transactions", "items", "basket"]

try:
    try:
        data = pd.read_csv(file_path)
    except Exception:
        data = pd.read_csv(file_path, delimiter=';')

    data.columns = data.columns.str.strip().str.lower()

    target_col = next((col for col in possible_cols if col in data.columns), None)
    if target_col is None:
        raise KeyError("❌ No valid transaction column found in this file.")

    transactions = [
        str(t).replace(" ", "").split(",")
        for t in data[target_col]
        if pd.notna(t)
    ]

except FileNotFoundError:
    raise FileNotFoundError(f"❌ File not found: {file_path}")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading data: {e}")

# ==========================================================
# STEP 2: Collect all the unique items for Brute force
# ==========================================================
all_items = sorted(set(item for sublist in transactions for item in sublist))

# ==========================================================
# STEP 3: User Input for Support & Confidence
# ==========================================================
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
# STEP 4: Brute-force Frequent Itemset Mining (with timing)
# ==========================================================
def get_support(itemset, transactions):
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

print("\nRunning Brute-Force Algorithm...")
start_brute = time.time()
frequent_itemsets_brute = brute_force_mining(transactions, min_support)
rules_brute = []
rules_brute_start = time.time()

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

rules_brute = generate_rules(frequent_itemsets_brute, min_confidence, transactions)
end_brute = time.time()
brute_force_time = end_brute - start_brute

# ==========================================================
# STEP 6: One-Hot Encoding for MLXTEND
# ==========================================================
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
one_hot = pd.DataFrame(te_ary, columns=te.columns_)

# ==========================================================
# STEP 7: Run Apriori and FP-Growth (with timing)
# ==========================================================
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("\nRunning Apriori Algorithm...")
start_apriori = time.time()
frequent_itemsets_ap = apriori(one_hot, min_support=min_support, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=min_confidence)
rules_ap = rules_ap.dropna()
rules_ap = rules_ap[(rules_ap['support'] > 0) & (rules_ap['confidence'] > 0)]
end_apriori = time.time()
apriori_time = end_apriori - start_apriori

print("Running FP-Growth Algorithm...")
start_fp = time.time()
frequent_itemsets_fp = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)
rules_fp = rules_fp.dropna()
rules_fp = rules_fp[(rules_fp['support'] > 0) & (rules_fp['confidence'] > 0)]
end_fp = time.time()
fp_growth_time = end_fp - start_fp

# ==========================================================
# STEP 8: Display Association Rules (All Algorithms)
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

# ==========================================================
# STEP 9: Display Timing Summary
# ==========================================================
print("\n\n================================================")
print("⏱️ EXECUTION TIME SUMMARY (seconds)")
print("================================================")
print(f"Brute-Force Algorithm: {brute_force_time:.4f} sec")
print(f"Apriori Algorithm:     {apriori_time:.4f} sec")
print(f"FP-Growth Algorithm:   {fp_growth_time:.4f} sec")
print("================================================")

print("\n✅ All algorithms executed successfully!")
