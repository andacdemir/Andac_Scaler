import numpy as np

# Scales the given input array of 2d to the desired mean and std along the axis
# specified.
# axis can be 0 or 1
# If axis=0, transform each feature (columnwise in the dataset)
# If axis=1, transform each sample (rowwise in the dataset)

class scaler(): 
    def __init__(self, input_array, mean, std, axis):
        # to convert the integer array to a floating array add 0.0
        self.inp = input_array + 0.0
        self.ax = axis
        self.mean1 = np.mean(self.inp, axis=self.ax)
        self.std1 = np.std(self.inp, axis=self.ax)
        self.mean2 = mean
        self.std2 = std
    
    def StandardScaler(self):
        if self.ax == 0: 
            for i, column in enumerate(self.inp.T):
                column = (column - self.mean1[i]) * self.std2 / self.std1[i] \
                         + self.mean2
                self.inp[:, i] = column

        elif self.ax == 1:
            for i, row in enumerate(self.inp):
                row = (row - self.mean1[i]) * self.std2 / self.std1[i] \
                      + self.mean2
                self.inp[i, :] = row

        return self.inp

    # You call the InverseScaler after calling the StandardScaler function
    # to scale the output of the StandardScaler function back to its 
    # original statistical properties 
    def InverseScaler(self):
        if self.ax == 0: 
            for i, column in enumerate(self.inp.T):
                column = (column - self.mean2) * self.std1[i] / self.std2 \
                         + self.mean1[i]
                self.inp[:, i] = column

        elif self.ax == 1:
            for i, row in enumerate(self.inp):
                row = (row - self.mean2) * self.std1[i] / self.std2 + \
                      + self.mean1[i]
                self.inp[i, :] = row

        return self.inp


# test the class
def main():
    # Test 1
    # Test with one dimension:
    x = np.array([[5,7,3],[3,2,2],[1,8,1],[4,0,9]])
    mean = 0
    std = 1
    Andac_scaler = scaler(x, mean, std, axis=0)   
    print(Andac_scaler.StandardScaler())
    print(Andac_scaler.InverseScaler())
    # Test 2
    # Test with both dimensions:
    Andac_scaler_0 = scaler(x, mean, std, axis=0) 
    scaled0 = Andac_scaler_0.StandardScaler() 
    Andac_scaler_1 = scaler(scaled0, mean, std, axis=1) 
    scaled1 = Andac_scaler_1.StandardScaler() 
    print(scaled0)
    print(scaled1)
    # Test 3:
    # Inverse transformations:
    scaled1_inversed = Andac_scaler_1.InverseScaler()
    scaled0_inversed = Andac_scaler_0.InverseScaler()
    print(scaled1_inversed)
    print(scaled0_inversed)

if __name__ == '__main__':
    main()

