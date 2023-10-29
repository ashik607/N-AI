import pandas as pd
import numpy as np
from scipy import stats
# Read data from CSV file into a DataFrame
df = pd.read_csv('E:\MM.csv')

# Output DataFrame
print(df)



# Calculating Mean, Median, Mode, Variance, and Standard Deviation
dataset = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# dataset = df['E:\B.csv'].tolist()
mean_value = np.mean(dataset)
median_value = np.median(dataset)
mode_value = stats.mode(dataset, keepdims=True).mode[0]  # Explicitly set keepdims parameter
variance_value = np.var(dataset)
std_deviation_value = np.std(dataset)

print("Mean of the dataset is:", mean_value)
print("Median of the dataset is:", median_value)
print("Mode of the dataset is:", mode_value)
print("Variance of the dataset is:", variance_value)
print("Standard Deviation of the dataset is:", std_deviation_value)




#    Unnamed: 0     Name  Age Gender  Marks
# 0           0   RAHIMA   22      F   85.0
# 1           1  RAHATUL   23      M   90.0
# 2           2   NAYEEM   24      M    NaN
# 3           3    SUHAN   25      M   70.0
# 4           4    IMRAN   22      M   82.0
# 5           5  SHADHIN   23      M    NaN
# 6           6    BAPPY   21      M   86.0