# Optimizers Mini Python Library

A simple implementation of optimization algorithms in Python, specifically Stochastic Gradient Descent (SGD). This library includes multiple loss functions, such as Mean Squared Error (MSE) and Cross-Entropy (CE), to help train machine learning models.

## Features
- Implements SGD for optimization.
- Supports Mean Squared Error (MSE) and Cross-Entropy (CE) loss functions.
- Utilizes NumPy for numerical operations.

## Installation

Clone this repository and install the dependencies using pip:

```bash
git clone https://github.com/Pranat1729/Optimizers-mini-python-library.git
cd Optimizers-mini-python-library
pip install -r requirements.txt
from optimizers import SGD
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
```
```python
# Load dataset
X, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = SGD(x_train, y_train, np.random.randn(x_train.shape[1]), epochs=100, loss='MSE')
model.check()
## Future Enhancements

- **Additional Optimization Algorithms**: Implement more optimization techniques such as Adam, RMSProp, and AdaGrad.
- **Extended Loss Functions**: Add support for other loss functions like Hinge Loss, Kullback-Leibler Divergence, etc.
- **Improved Logging**: Integrate advanced logging mechanisms for better tracking of model performance during training.
- **Parallelization**: Implement parallel training for larger datasets to enhance performance.
```
## Contact

For any questions or suggestions, feel free to reach out to me via GitHub Issues or email:

- GitHub: [https://github.com/Pranat1729](https://github.com/Pranat1729)
- Email: pranat1729@gmail.com
- Linkedin: [Pranat's Linkedin](https://www.linkedin.com/in/pranat-sharma-a55a77168/)
