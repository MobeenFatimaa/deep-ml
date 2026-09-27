import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    
    total_count = len(labels)
    counts = Counter(labels)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total_count
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    parent_labels = [ex[target_attr] for ex in examples]
    parent_entropy = calculate_entropy(parent_labels)
    
    total_count = len(examples)
    
    attr_subsets = {}
    for ex in examples:
        val = ex[attr]
        attr_subsets.setdefault(val, []).append(ex[target_attr])
        
    weighted_child_entropy = 0.0
    for subset_labels in attr_subsets.values():
        weight = len(subset_labels) / total_count
        weighted_child_entropy += weight * calculate_entropy(subset_labels)
        
    return parent_entropy - weighted_child_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [ex[target_attr] for ex in examples]
    counts = Counter(labels)
    
    # Sort by frequency descending (-count), then label alphabetically
    sorted_labels = sorted(counts.keys(), key=lambda label: (-counts[label], label))
    return sorted_labels[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict | str:
    """Build a decision tree using the ID3 algorithm."""
    # Base Case 1: All examples have the same target value
    first_target = examples[0][target_attr]
    if all(ex[target_attr] == first_target for ex in examples):
        return first_target

    # Base Case 2: Attributes list is empty
    if not attributes:
        return majority_class(examples, target_attr)

    # Find the best attribute to split on
    best_attr = None
    best_gain = -1.0

    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        # Tie-breaking rule: strict > keeps the first attribute in attributes list on ties
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}
    remaining_attributes = [a for a in attributes if a != best_attr]

    # Process attribute values in sorted order
    unique_values = sorted(list({ex[best_attr] for ex in examples}))

    for val in unique_values:
        sub_examples = [ex for ex in examples if ex[best_attr] == val]
        
        if not sub_examples:
            # If subset is empty, assign the majority class of current parent set
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            # Recurse on remaining attributes
            subtree = learn_decision_tree(sub_examples, remaining_attributes, target_attr)
            tree[best_attr][val] = subtree

    return tree