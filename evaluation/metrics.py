from collections import defaultdict


def compute_metrics(y_true, y_pred):

    classes = sorted(set(y_true))

    results = {}

    total = len(y_true)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    accuracy = correct / total if total else 0

    results["accuracy"] = accuracy


    precision = {}
    recall = {}
    f1 = {}


    for c in classes:

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)

        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)

        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)


        p = tp / (tp + fp) if (tp + fp) else 0

        r = tp / (tp + fn) if (tp + fn) else 0

        f = 2 * p * r / (p + r) if (p + r) else 0


        precision[c] = p
        recall[c] = r
        f1[c] = f


    macro_f1 = sum(f1.values()) / len(f1) if f1 else 0


    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    results["macro_f1"] = macro_f1


    return results