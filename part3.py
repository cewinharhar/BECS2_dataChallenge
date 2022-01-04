"""
## Part 3: Data Preparation
In this section the raw data is prepared and reshaped to be fed into the different models. Furhtermore, the distribution of the input data is visualized to check if the data set is balanced. The data is converted into two main variables X (patiens and the coresponding protein quantities) and y (patients and the coresponding healt condition).

"""
import pandas as pd

#create pandas dataframe
path = "tidy.csv"
pathMet = "metadata.csv"
tidy = pd.read_csv(path, sep=",")
tidyMet = pd.read_csv(pathMet, sep=";", index_col=0)

#remove samples which are not in the metadata index column (quality controle etc)
tidy = tidy[ (tidy["R.FileName"].isin(tidyMet.index)) ]
tidyMer    = pd.merge(tidy, tidyMet, how="left", on="R.FileName")
tidySub = tidyMer[["R.FileName", "uniprot", "meanAbu", "Cancer"]]
tidySub.Cancer.value_counts()


#reshape data for model
#X data
tidyReshaped = tidySub.pivot(index = "R.FileName", columns = "uniprot", values = "meanAbu")
tidyReshaped.head()

#y condition
Group =  tidySub.drop(["uniprot", "meanAbu"], axis=1)
Group = Group.drop_duplicates().reset_index(drop=True)
Group.head()

#merge X and y and set dataframe to numerical values
data = pd.merge(tidyReshaped, Group, how="left", on="R.FileName")
data = data.set_index("R.FileName")

X_ = data.iloc[:, :-1].apply(np.log2)
y_ = data.iloc[:,-1]

#check first 10 entries of the dataframe 
data[:10]

fig, ax = plt.subplots(1,1, figsize=(30, 5))

ax.boxplot(X_)