import numpy as np
target=np.genfromtxt('test_targets.csv')
prediction=np.genfromtxt('my_predictions.csv')

print(sum(target==prediction)/prediction.shape[0])
