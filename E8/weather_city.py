import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    # Load the input files specified as command-line arguments
    labelled_data_path = sys.argv[1]
    unlabelled_data_path = sys.argv[2]
    labelled_data = pd.read_csv(labelled_data_path)
    unlabelled_data = pd.read_csv(unlabelled_data_path)

    # Extract features and target labels from the labelled data
    features = labelled_data.iloc[:, 2:].values
    labels = labelled_data.iloc[:, 0].values

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Construct a RandomForest model with feature scaling
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=3, random_state=42))
    ])
    
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Calculate and print the accuracy score on the validation data
    accuracy = model.score(X_val, y_val)
    print(f"Model's validation Accuracy: {accuracy}")

    # Check incorrect predictions
    # df = pd.DataFrame({'truth': y_val, 'prediction': model.predict(X_val)})
    # print(df[df['truth'] != df['prediction']])

    # Prepare the unlabelled data for predictions
    unlabelled_features = unlabelled_data.iloc[:, 2:].values
    predictions = model.predict(unlabelled_features)

    # Output the predictions
    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)

if __name__ == "__main__":
    main()
