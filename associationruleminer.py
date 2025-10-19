# ----------------------------------------------------------
# Market Basket Analysis: Brute Force, Apriori, FP-Growth
# ----------------------------------------------------------
# Author: Taymar Walters
# Description:
#   1. Load and parse a transactions CSV file (robust to header differences).
#   2. Run a brute-force frequent itemset mining algorithm.
#   3. Generate association rules from brute-force results.
#   4. Run Apriori and FP-Growth using mlxtend for comparison.
# ----------------------------------------------------------

import itertools
import subprocess
import sys
import os
import pandas as pd

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
# STEP 1: Select and Load Dataset (with flexible parsing)
# ==========================================================
print("Here are the following transactional databases\n"
      " 1) Generic\n 2) Nike\n 3) Best Buy\n 4) Coffee Shop\n 5) K-mart\n ")
items = ""
def selectfile():
    while True:
        try:
            fileNumber = int(input("Enter number to select a database: \n"))
            match fileNumber:
                case 1:
                    items = "generic_items.csv"
                    return "generic_transactions.csv", items
                case 2:
                    items = "nike_products.csv"
                    return "nike_product_transactions.csv", items
                case 3:
                    items = "bestbuy_products.csv"
                    return "bestbuy_transactions.csv", items
                case 4:
                    items = "coffee_items.csv"
                    return "coffee_transactions.csv", items
                case 5:
                    items = "k-mart_items.csv"
                    return "k-mart_transactions.csv", items
                case _:
                    print("Invalid input. Please try again.")
        except ValueError:
            print("Please enter a valid number between 1–5.")

transactions, items = selectfile()  
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, items)

print("================================================")
print("Here are the list of items corresponding to the transactions:")
print("================================================")
df = pd.read_csv(file_path)
df = df.dropna(how='all')
df.columns = df.columns.str.strip()
df["Item #"] = df["Item #"].astype(int)
print(df.to_string(index=False))
print("================================================")
# Get file path
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, transactions)

# Columns that might represent transactions
possible_cols = ["transaction", "transactions", "items", "basket"]

try:
    # Try to read the CSV normally, then fallback to alternate delimiter
    try:
        data = pd.read_csv(file_path)
    except Exception:
        data = pd.read_csv(file_path, delimiter=';')

    data.columns = data.columns.str.strip().str.lower()
    #print(f"\n✅ File loaded successfully: {os.path.basename(file_path)}")
    #print(f"Columns detected: {list(data.columns)}\n")

    # Detect transaction column dynamically
    target_col = None
    for col in possible_cols:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        raise KeyError("❌ No valid transaction column found in this file.")

    transactions = [
        str(t).replace(" ", "").split(",")
        for t in data[target_col]
        if pd.notna(t)
    ]

    #print(f"📦 Loaded {len(transactions)} transactions.\n")

except FileNotFoundError:
    raise FileNotFoundError(f"❌ File not found: {file_path}")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading data: {e}")

# ==========================================================
# STEP 2: Collect all unique items
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
# STEP 4: Brute-force Frequent Itemset Mining
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

frequent_itemsets_brute = brute_force_mining(transactions, min_support)

# ==========================================================
# STEP 5: Generate Association Rules (Brute Force)
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

rules_brute = generate_rules(frequent_itemsets_brute, min_confidence, transactions)

# ==========================================================
# STEP 6: One-Hot Encoding for MLXTEND
# ==========================================================
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
one_hot = pd.DataFrame(te_ary, columns=te.columns_)

# ==========================================================
# STEP 7: Run Apriori and FP-Growth
# ==========================================================
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Apriori
frequent_itemsets_ap = apriori(one_hot, min_support=min_support, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=min_confidence)
rules_ap = rules_ap.dropna()
rules_ap = rules_ap[(rules_ap['support'] > 0) & (rules_ap['confidence'] > 0)]

# FP-Growth
frequent_itemsets_fp = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)
rules_fp = rules_fp.dropna()
rules_fp = rules_fp[(rules_fp['support'] > 0) & (rules_fp['confidence'] > 0)]


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

print("\n✅ All algorithms executed successfully!")
