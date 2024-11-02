import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Output template for results
OUTPUT_TEMPLATE = (
    'Bayesian classifier:     {bayes_rgb:.3f}  {bayes_convert:.3f}\n'
    'kNN classifier:          {knn_rgb:.3f}  {knn_convert:.3f}\n'
    'Rand forest classifier:  {rf_rgb:.3f}  {rf_convert:.3f}\n'
)

# Dictionary to map colour names to RGB values
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 112, 0),
    'yellow': (255, 255, 0),
    'green': (0, 231, 0),
    'blue': (0, 0, 255),
    'purple': (185, 0, 185),
    'brown': (117, 60, 0),
    'pink': (255, 184, 184),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])

def plot_predictions(model, lum=67, resolution=300):
    """
    Generate and plot model predictions over a slice of LAB colour space.
    """
    wid, hei = resolution, resolution
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)
    X_grid = lab2rgb(lab_grid)
    y_grid = model.predict(X_grid.reshape((-1, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    plt.figure(figsize=(10, 5))
    plt.suptitle(f'Predictions at L={lum}')
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.imshow(X_grid.reshape((hei, wid, -1)))
    plt.xlabel('A')
    plt.ylabel('B')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.imshow(pixels)
    plt.xlabel('A')

def main():
    # Load and split data
    data = pd.read_csv(sys.argv[1])
    X = data[['R', 'G', 'B']].values / 255.0
    y = data['Label'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Bayesian classifier (RGB)
    bayes_rgb_model = GaussianNB()
    bayes_rgb_model.fit(X_train, y_train)
    bayes_rgb_score = bayes_rgb_model.score(X_valid, y_valid)

    # Train and evaluate Bayesian classifier (converted to LAB)
    bayes_convert_model = make_pipeline(FunctionTransformer(rgb2lab, validate=False), GaussianNB())
    bayes_convert_model.fit(X_train, y_train)
    bayes_convert_score = bayes_convert_model.score(X_valid, y_valid)

    # Train and evaluate k-Nearest Neighbors classifier (RGB)
    knn_rgb_model = KNeighborsClassifier(n_neighbors=8)
    knn_rgb_model.fit(X_train, y_train)
    knn_rgb_score = knn_rgb_model.score(X_valid, y_valid)

    # Train and evaluate k-Nearest Neighbors classifier (converted to LAB)
    knn_convert_model = make_pipeline(FunctionTransformer(rgb2lab, validate=False), KNeighborsClassifier(n_neighbors=8))
    knn_convert_model.fit(X_train, y_train)
    knn_convert_score = knn_convert_model.score(X_valid, y_valid)

    # Train and evaluate Random Forest classifier (RGB)
    rf_rgb_model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=6)
    rf_rgb_model.fit(X_train, y_train)
    rf_rgb_score = rf_rgb_model.score(X_valid, y_valid)

    # Train and evaluate Random Forest classifier (converted to LAB)
    rf_convert_model = make_pipeline(FunctionTransformer(rgb2lab, validate=False), RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=6))
    rf_convert_model.fit(X_train, y_train)
    rf_convert_score = rf_convert_model.score(X_valid, y_valid)

    # Output results
    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=bayes_rgb_score,
        bayes_convert=bayes_convert_score,
        knn_rgb=knn_rgb_score,
        knn_convert=knn_convert_score,
        rf_rgb=rf_rgb_score,
        rf_convert=rf_convert_score
    ))

    # Generate and save prediction plots
    plot_predictions(bayes_rgb_model)
    plt.savefig('predictions-bayes-rgb.png')
    plt.close()

    plot_predictions(bayes_convert_model)
    plt.savefig('predictions-bayes-convert.png')
    plt.close()

    plot_predictions(knn_rgb_model)
    plt.savefig('predictions-knn-rgb.png')
    plt.close()

    plot_predictions(knn_convert_model)
    plt.savefig('predictions-knn-convert.png')
    plt.close()

    plot_predictions(rf_rgb_model)
    plt.savefig('predictions-rf-rgb.png')
    plt.close()

    plot_predictions(rf_convert_model)
    plt.savefig('predictions-rf-convert.png')
    plt.close()

if __name__ == '__main__':
    main()
