# import necessary libraries
import pandas as pd
import numpy as np

# create a dummy array
arr = np.load(sys.argv[1]) #"C:\\Users\\user\\Desktop\\topDetectionGS.npy")

# display the array
#print(arr)

# convert array into dataframe
DF = pd.DataFrame(arr)

# save the dataframe as a csv file
DF.to_csv(sys.argv[1][:-4] + "\\.csv") #"C:\\Users\\user\\Desktop\\GS.csv")