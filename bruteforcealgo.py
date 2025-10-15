# ----------------------------------------------------------
# Brute-force frequent itemset mining and rule generation
# (NO external libraries — pure Python)
# ----------------------------------------------------------

# STEP 1: Read and parse the dataset manually
filename = r"C:\SCHOOL PROJECTS\Data Mining\walters_taymar_midtermproj\generic_transactions.csv"


# Read file
with open(filename, "r") as f:
    lines = f.readlines()

# Skip header and parse transactions
transactions = []
for line in lines[1:]:
    parts = line.strip().split(",")
    # Last column may have multiple items
    items = [p.strip() for p in parts[1:]] if len(parts) > 2 else [p.strip() for p in parts[1].split()]
    # Clean up
    tx = []
    for chunk in parts[1:]:
        for item in chunk.split(","):
            item = item.strip()
            if item:
                tx.append(item)
    transactions.append(tx)

# Collect all unique items
unique_items = []
for t in transactions:
    for item in t:
        if item not in unique_items:
            unique_items.append(item)

unique_items.sort()


# ----------------------------------------------------------
# STEP 2: Helper functions
# ----------------------------------------------------------

def get_support(itemset, transactions):
    """Compute the support (fraction) of an itemset."""
    count = 0
    for t in transactions:
        if all(i in t for i in itemset):
            count += 1
    return count / len(transactions)


def generate_combinations(items, k):
    """Manual version of itertools.combinations."""
    results = []

    def helper(start, combo):
        if len(combo) == k:
            results.append(combo[:])
            return
        for i in range(start, len(items)):
            combo.append(items[i])
            helper(i + 1, combo)
            combo.pop()

    helper(0, [])
    return results


# ----------------------------------------------------------
# STEP 3: Brute-force frequent itemset mining
# ----------------------------------------------------------

def brute_force_frequent_itemsets(transactions, unique_items, min_support):
    all_frequents = {}
    k = 1

    while True:
        # Generate all k-itemsets manually
        candidates = generate_combinations(unique_items, k)
        frequent_k_itemsets = {}

        for itemset in candidates:
            support = get_support(itemset, transactions)
            if support >= min_support:
                frequent_k_itemsets[tuple(itemset)] = support

        if not frequent_k_itemsets:
            print(f"No frequent {k}-itemsets found. Terminating.")
            break

        print(f"Found {len(frequent_k_itemsets)} frequent {k}-itemsets.")
        all_frequents.update(frequent_k_itemsets)
        k += 1

    return all_frequents


# ----------------------------------------------------------
# STEP 4: Generate association rules manually
# ----------------------------------------------------------

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    keys = list(frequent_itemsets.keys())

    for itemset in keys:
        if len(itemset) > 1:
            # Generate all possible non-empty proper subsets
            n = len(itemset)
            subsets = []
            for i in range(1, n):
                subsets.extend(generate_combinations(list(itemset), i))

            for antecedent in subsets:
                consequent = [x for x in itemset if x not in antecedent]
                if not consequent:
                    continue

                support_itemset = frequent_itemsets[itemset]
                support_antecedent = frequent_itemsets.get(tuple(sorted(antecedent)), None)

                if support_antecedent:
                    confidence = support_itemset / support_antecedent
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support_itemset, confidence))
    return rules


# ----------------------------------------------------------
# STEP 5: Run algorithm
# ----------------------------------------------------------

min_support = 0.3
min_confidence = 0.6

frequent_itemsets = brute_force_frequent_itemsets(transactions, unique_items, min_support)
association_rules = generate_association_rules(frequent_itemsets, min_confidence)


# ----------------------------------------------------------
# STEP 6: Display results
# ----------------------------------------------------------

print("\n=== FREQUENT ITEMSETS ===")
for itemset, support in frequent_itemsets.items():
    print(f"{itemset}: {support:.2f}")

print("\n=== ASSOCIATION RULES ===")
for rule in association_rules:
    antecedent, consequent, support, confidence = rule
    print(f"{antecedent} -> {consequent} (Support: {support:.2f}, Confidence: {confidence:.2f})")
