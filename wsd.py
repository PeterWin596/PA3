import sys
import math
from collections import defaultdict

def read_train_data(train_file):
    """Reads the training data and extracts labeled contexts."""
    data = []
    with open(train_file, 'r') as f:
        for line in f:
            if '<instance id=' in line:
                instance_id = line.split('"')[1]
            if '<answer instance=' in line:
                sense = line.split('senseid="')[1].split('"')[0]
            if '<context>' in line:
                context = []
                for cline in f:
                    if '</context>' in cline:
                        break
                    context.extend(cline.strip().split())
                data.append((instance_id, context, sense))
    return data

def extract_features(data):
    """Extracts word-based features and computes log-likelihood ratios."""
    word_counts = defaultdict(lambda: {'phone': 0, 'product': 0})
    sense_counts = {'phone': 0, 'product': 0}

    for _, words, sense in data:
        sense_counts[sense] += 1
        for word in set(words):  # Use set to avoid duplicate words in the same instance
            word_counts[word][sense] += 1

    decision_list = []
    for word, counts in word_counts.items():
        phone_prob = (counts['phone'] + 1) / (sense_counts['phone'] + 2)  # Add-one smoothing
        product_prob = (counts['product'] + 1) / (sense_counts['product'] + 2)
        log_likelihood = abs(math.log(phone_prob / product_prob))
        best_sense = 'phone' if phone_prob > product_prob else 'product'
        decision_list.append((log_likelihood, word, best_sense))
    
    decision_list.sort(reverse=True, key=lambda x: x[0])
    return decision_list

def save_model(decision_list, model_file):
    """Saves the decision list model."""
    with open(model_file, 'w') as f:
        for log_likelihood, word, sense in decision_list:
            f.write(f"{word}\t{log_likelihood:.4f}\t{sense}\n")

def classify_test_data(test_file, decision_list):
    """Classifies test data using the decision list."""
    predictions = []
    with open(test_file, 'r') as f:
        for line in f:
            if '<instance id=' in line:
                instance_id = line.split('"')[1]
            if '<context>' in line:
                context = []
                for cline in f:
                    if '</context>' in cline:
                        break
                    context.extend(cline.strip().split())
                predicted_sense = "phone"  # Default sense
                for _, word, sense in decision_list:
                    if word in context:
                        predicted_sense = sense
                        break
                predictions.append(f'<answer instance="{instance_id}" senseid="{predicted_sense}"/>')
    
    for prediction in predictions:
        print(prediction)

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt")
        sys.exit(1)

    train_file, test_file, model_file = sys.argv[1], sys.argv[2], sys.argv[3]
    training_data = read_train_data(train_file)
    decision_list = extract_features(training_data)
    save_model(decision_list, model_file)
    classify_test_data(test_file, decision_list)

if __name__ == "__main__":
    main()
