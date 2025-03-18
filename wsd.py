import sys
import re
import math
from collections import defaultdict, Counter

# Function to extract context words (features)
def extract_features(sentence, target_word="line"):
    words = sentence.lower().split()
    if target_word in words:
        index = words.index(target_word)
        return words[max(0, index - 2): index] + words[index + 1: index + 3]
    return []

# Train Decision List from training file
def train_decision_list(train_file):
    sense_counts = Counter()
    feature_sense_counts = defaultdict(lambda: Counter())
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'(.+)<senseid="(.*?)"', line)
            if match:
                sentence, sense = match.groups()
                features = extract_features(sentence)
                sense_counts[sense] += 1
                for feature in features:
                    feature_sense_counts[feature][sense] += 1
    
    decision_list = []
    for feature, senses in feature_sense_counts.items():
        if len(senses) > 1:
            best_sense, second_sense = senses.most_common(2)
            log_likelihood = abs(math.log((best_sense[1] + 1) / (second_sense[1] + 1)))
        else:
            best_sense = senses.most_common(1)[0]
            log_likelihood = math.log(best_sense[1] + 1)
        decision_list.append((log_likelihood, feature, best_sense[0]))
    
    decision_list.sort(reverse=True, key=lambda x: x[0])
    return decision_list

# Apply Decision List to classify test sentences
def classify_test(decision_list, test_file):
    results = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            features = extract_features(line)
            prediction = "phone"  # Default sense
            for _, feature, sense in decision_list:
                if feature in features:
                    prediction = sense
                    break
            results.append(f"{line.strip()} <answer senseid=\"{prediction}\" />")
    
    return results

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    
    decision_list = train_decision_list(train_file)
    
    with open(model_file, 'w', encoding='utf-8') as f:
        for log_likelihood, feature, sense in decision_list:
            f.write(f"{feature}\t{log_likelihood}\t{sense}\n")
    
    predictions = classify_test(decision_list, test_file)
    for prediction in predictions:
        print(prediction)
