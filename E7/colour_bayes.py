import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skimage.color import rgb2lab
from colour_bayes_hint import plot_predictions

def main(infile):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(infile)
    
    # Extract the RGB features and the corresponding colour labels
    X = data[['R', 'G', 'B']].values / 255.0  # Scale RGB values to the range 0-1
    y = data['Label'].values

    # Split the dataset into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naïve Bayes classifier using the RGB data
    model_rgb = GaussianNB()
    model_rgb.fit(X_train, y_train)

    # Evaluate the model on the validation dataset
    y_pred_rgb = model_rgb.predict(X_val)
    accuracy_rgb = accuracy_score(y_val, y_pred_rgb)
    print(f'Accuracy (RGB model): {accuracy_rgb:.2f}')

    # Generate and save the plot for predictions using the RGB model
    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plt.show()

    # Convert the RGB features to LAB colour space for LAB-based model
    X_lab = rgb2lab(X.reshape(-1, 1, 3)).reshape(-1, 3)

    # Split the LAB data into training and validation subsets
    X_train_lab, X_val_lab, _, _ = train_test_split(X_lab, y, test_size=0.2, random_state=42)

    # Train a Naïve Bayes classifier using the LAB data
    model_lab = GaussianNB()
    model_lab.fit(X_train_lab, y_train)

    # Evaluate the LAB-based model on the validation data
    y_pred_lab = model_lab.predict(X_val_lab)
    accuracy_lab = accuracy_score(y_val, y_pred_lab)
    print(f'Accuracy (LAB model): {accuracy_lab:.2f}')

    # Generate and save the plot for predictions using the LAB model
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python colour_bayes.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])
