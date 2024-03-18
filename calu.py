def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, f1_score

if __name__ == "__main__":
    tp = 436
    fn = 77
    fp = 139
    tn = 348
    accuracy, f1_score = calculate_metrics(tp, tn, fp, fn)
    print(f"Accuracy: {accuracy}, F1 Score: {f1_score}")