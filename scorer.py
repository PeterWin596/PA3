import sys
from collections import defaultdict

def read_answers(file):
    """Reads the answer file and returns a dictionary {instance_id: sense}."""
    answers = {}
    with open(file, 'r') as f:
        for line in f:
            if "senseid=" in line:
                instance = line.split("instance=\"")[1].split("\"")[0]
                sense = line.split("senseid=\"")[1].split("\"")[0]
                answers[instance] = sense
    return answers

def compute_accuracy(predictions, gold_standard):
    """Computes accuracy and confusion matrix."""
    correct = 0
    total = len(gold_standard)
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for instance, true_sense in gold_standard.items():
        pred_sense = predictions.get(instance, "UNKNOWN")
        if pred_sense == true_sense:
            correct += 1
        confusion_matrix[true_sense][pred_sense] += 1

    accuracy = correct / total * 100
    return accuracy, confusion_matrix

def print_confusion_matrix(confusion_matrix):
    """Prints confusion matrix."""
    senses = sorted(set(confusion_matrix.keys()) | set(k for v in confusion_matrix.values() for k in v))
    print("\nConfusion Matrix:")
    print("\t" + "\t".join(senses))
    for true_sense in senses:
        row = [str(confusion_matrix[true_sense][pred_sense]) for pred_sense in senses]
        print(f"{true_sense}\t" + "\t".join(row))

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scorer.py my-line-answers.txt line-key.txt")
        sys.exit(1)

    pred_file, key_file = sys.argv[1], sys.argv[2]
    predictions = read_answers(pred_file)
    gold_standard = read_answers(key_file)

    accuracy, confusion_matrix = compute_accuracy(predictions, gold_standard)

    print(f"Accuracy: {accuracy:.2f}%")
    print_confusion_matrix(confusion_matrix)

if __name__ == "__main__":
    main()
