import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
​
import seaborn as sns
import pandas as pd
data_main = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Raw-Data.csv')
data_main.shape
data_main.head(10)