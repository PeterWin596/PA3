import sys
from collections import Counter

# Function to load answers from a file
def load_answers(file_path):
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "<answer senseid=" in line:
                sense = line.split('<answer senseid="')[1].split('"')[0]
                answers.append(sense)
    return answers

if __name__ == "__main__":
    predicted_file = sys.argv[1]
    key_file = sys.argv[2]
    
    predicted_senses = load_answers(predicted_file)
    key_senses = load_answers(key_file)
    
    if len(predicted_senses) != len(key_senses):
        print("Error: Mismatch in number of lines between predicted and key files.")
        sys.exit(1)
    
    total = len(key_senses)
    correct = sum(p == k for p, k in zip(predicted_senses, key_senses))
    accuracy = correct / total * 100
    
    # Confusion matrix
    confusion_matrix = Counter()
    unique_senses = set(key_senses + predicted_senses)
    
    for p, k in zip(predicted_senses, key_senses):
        confusion_matrix[(k, p)] += 1
    
    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    for k in unique_senses:
        for p in unique_senses:
            print(f"{k} -> {p}: {confusion_matrix[(k, p)]}")
