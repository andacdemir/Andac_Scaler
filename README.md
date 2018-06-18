# Andac_Scaler 

Scales the given 2d input array to the desired mean and std. along the axis specified.

Axis can be 0 or 1 which cannot be handled by sklearn.preprocessing.StandardScaler 

If axis=0, transform each feature (column-wise in the dataset)

If axis=1, transform each sample (row-wise in the dataset)
