import numpy as np;
import matplotlib.pyplot as plt;

xs = np.array([1, 2, 3, 4, 5, 6]);
ys = np.array([1, 2.5, 2.5, 4.7, 5, 6.2]);

m, b = 0, 0 #slope, intercept of fit line
alpha = 0.001 #learning constant

for i in range(0, 100):
    
    #print out the MSE for each iteration.
    mse = np.mean((xs*m + b - ys)**2)
    print(f"iteration {i}: {mse}")

    #update slope (m) and intercept (b) for fit line.
    m = m - alpha * (-2/len(xs)) * np.sum(xs * (ys - (m*xs + b)));
    b = b - alpha * (-2/len(xs)) * np.sum(ys - m*xs + b);

#draw the regression line
x_line = np.array([xs.min(), xs.max()])
y_line = m * x_line + b;

plt.plot(x_line, y_line, color = 'green');
plt.scatter(xs, ys);
plt.show();
