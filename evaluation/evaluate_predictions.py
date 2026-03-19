import json

from evaluation.metrics import compute_metrics


def normalize(label):

    label = label.lower().strip()

    mapping = {
        "supported": "supported",
        "refuted": "refuted",
        "not enough evidence": "not enough evidence",
        "conflicting evidence/cherrypicking": "conflicting",
        "conflicting": "conflicting"
    }

    return mapping.get(label, label)


def evaluate(predictions_path, dataset_path):

    with open(predictions_path) as f:
        predictions = json.load(f)

    with open(dataset_path) as f:
        dataset = json.load(f)


    gold = {
        item["claim"]: normalize(item["label"])
        for item in dataset
    }


    y_true = []
    y_pred = []


    for pred in predictions:
        claim = pred["claim"]

        if claim not in gold:
            continue

        y_true.append(gold[claim])
        y_pred.append(normalize(pred["prediction"]))


    results = compute_metrics(y_true, y_pred)


    print("\n===== Evaluation =====\n")

    print("Accuracy:", round(results["accuracy"], 3))

    print("\nPer-class metrics:\n")


    for label in results["precision"]:

        p = results["precision"][label]
        r = results["recall"][label]
        f = results["f1"][label]

        print(label)

        print("  Precision:", round(p, 3))
        print("  Recall:   ", round(r, 3))
        print("  F1:       ", round(f, 3))

        print()


    print("Macro F1:", round(results["macro_f1"], 3))