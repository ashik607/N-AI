
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Read data from CSV file using pandas
# df = pd.read_csv('E:\SS.csv')   # Replace 'your_data.csv' with your actual CSV file name

# # Extract the features (x) and target (y) from the DataFrame
# x = np.array(df['A'])  # Replace 'Feature_Column' with the actual column name for features
# y = np.array(df['B'])   # Replace 'Target_Column' with the actual column name for target

# # Corresponding output data (target)
# # Calculate the mean of x and y
# mean_x = np.mean(x)
# mean_y = np.mean(y)
# n = np.array(df)
# print(n)
# siz = np.size(x)
# print("The size of the array:", siz)

# # Calculate slope b and y-intercept a using the formula
# # b = Σ(xi*yi)-n*(mean_x)(mean-y)/Σ(xi*x1)-n*(mean_x)^2
# # a = ȳ - b * x̄
# numerator = np.sum(x*y) - siz*mean_x*mean_y
# denominator = np.sum(x*x) - siz*mean_x*mean_x
# slope = numerator / denominator
# intercept = mean_y - slope * mean_x
# print("The numerator is :",numerator)
# print("The denominator is :",denominator)

# # Predict y values using the linear regression equation: y = mx + b
# predicted_y = slope * x + intercept

# # Plot the original data points
# plt.scatter(x, y, color='blue', label='Actual Data')

# # Plot the regression line
# plt.plot(x, predicted_y, color='red', label='Regression Line')

# # Add labels and legend
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Simple Linear Regression')
# plt.legend()

# # Show the plot
# plt.show()



# [[10 11]
#  [11 13]
#  [12 12]
#  [13 15]
#  [14 17]
#  [15 18]
#  [16 18]
#  [17 19]
#  [18 20]
#  [19 22]]
# The size of the array: 10
# The numerator is : 96.5
# The denominator is : 82.5

import numpy as np
import matplotlib.pyplot as plt

# Sample input data (feature)
x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

# Corresponding output data (target)
y = np.array([11, 13, 12, 15, 17, 18, 18, 19, 20, 22])

# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)
siz = np.size(x)
print("The size of the array:", siz)



# Calculate slope (m) and y-intercept (b) using the formula
# m = Σ((x_i - x̄)(y_i - ȳ)) / Σ((x_i - x̄)^2)
# b = ȳ - m * x̄
numerator = np.sum(x*y) - siz*mean_x*mean_y
denominator = np.sum(x*x) - siz*mean_x*mean_x
slope = numerator / denominator
intercept = mean_y - slope * mean_x
print("The numerator is :",numerator)
print("The denominator is :",denominator)

# Predict y values using the linear regression equation: y = mx + b
predicted_y = slope * x + intercept

# Plot the original data points
plt.scatter(x, y, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(x, predicted_y, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()

# Show the plot
plt.show()
