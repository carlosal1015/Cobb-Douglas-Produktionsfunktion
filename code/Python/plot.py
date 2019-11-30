import numpy as np
import matplotlib.pyplot as plt
X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict
plt.plot(X_new, y_predict, 'r--',X, y, 'b.')
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

