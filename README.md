# Fake News Detection Project

## Project Description
This project focuses on detecting fake news using various machine learning algorithms. The dataset includes news articles labeled as fake or not fake. The project explores different preprocessing techniques and machine learning models to achieve accurate detection.

## Installation and Setup
To run this project, you need to have Python and several libraries installed. You can install the required libraries using:

```sh
pip install -r requirements.txt
```

## Data Preprocessing
The data preprocessing steps include:

1. **Loading Data**: The dataset is loaded using `pandas.read_csv`.
2. **Handling Null Values**: Null values in the title and text columns are filled with placeholders.
3. **Merging Columns**: The title and text columns are merged into a single column.
4. **Visualizations**: Word clouds and histograms for fake and not fake news.
5. **Tokenization**: Text data is tokenized and converted to sequences.
6. **Normalization**: The tokenized data is scaled to be between 0 and 1.

## Algorithms and Models
### Back Propagation
Back propagation is a supervised learning algorithm used for training neural networks. It involves forward propagation of inputs, calculation of error, and backward propagation to adjust the weights.

### Bayesian Neural Network
Bayesian neural networks incorporate Bayesian inference for better uncertainty estimation in predictions. This involves using probability distributions rather than point estimates for network weights.

### Keras
Keras is a high-level neural networks API that simplifies building and training deep learning models. The notebook likely includes models built using the Keras library.

### Levenberg-Marquardt
The Levenberg-Marquardt algorithm is an optimization technique used to solve non-linear least squares problems, often applied in training neural networks.

### Resilient Propagation
Resilient propagation (Rprop) is a learning algorithm for neural networks that adapts the weight updates based on the error, improving convergence speed.

## Usage
To use this project, follow these steps:

1. Clone the repository.
2. Install the necessary libraries.
3. Run the Jupyter notebooks in the following order:
   - `Data Preprocessing.ipynb`
   - `Back Propgation.ipynb`
   - `BayesionNeuralNetwork.ipynb`
   - `Keras.ipynb`
   - `Levenberg.ipynb`
   - `ResilientPropagation.ipynb`

## Results
The results of the models, including accuracy, precision, recall, and F1 scores, will be displayed in the respective notebooks.

## Contributing
Contributions are welcome. Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.
