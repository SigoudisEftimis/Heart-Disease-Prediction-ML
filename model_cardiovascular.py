import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from tabulate import tabulate
import matplotlib.pyplot as plt
import pickle
import os

def load_data(file_path, delimiter=','):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter)
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


def preprocess_data(data):
    processed_data = data.copy()
    y = processed_data.pop('Heart_Disease')
    processed_data = label_encode_categorical_columns(processed_data)
    return processed_data, y

def label_encode_categorical_columns(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

def train_and_evaluate_model(models,model, algorithm, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    export_models(model, algorithm)
    return accuracy



def export_models(trained_model, algorithm):
    folder_path = "training_models"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, algorithm + '.pkl')
    with open(file_path, 'wb') as model_file:
        pickle.dump(trained_model, model_file)


if __name__ == "__main__":
    file_path = "cardio_dataset.csv"
    data = load_data(file_path)

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_dict = {
        "LogReg": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis()
    }

    results = []
    models = []

    for algorithm_choice, model in model_dict.items():
        accuracy = train_and_evaluate_model(models ,model , algorithm_choice, X_train, y_train, X_test, y_test)
        results.append((algorithm_choice, accuracy))

    # Sort results by accuracy in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    table = tabulate(results, headers=["Algorithm", "Accuracy"], tablefmt='grid')
    print("Accuracy results from best to worst:")
    print(table)
    

    # Display results as a bar plot
    algorithms = [result[0] for result in results]
    accuracies = [result[1] for result in results]

    
    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(algorithms, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Algorithms')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True)  
    plt.show()
    
    

