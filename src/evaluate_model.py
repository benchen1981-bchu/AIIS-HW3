from sklearn.metrics import classification_report, confusion_matrix
import json
import os

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        'classification_report': report,
        'confusion_matrix': cm
    }

    os.makedirs("../reports", exist_ok=True)
    with open("../reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Metrics saved.")
    return metrics
