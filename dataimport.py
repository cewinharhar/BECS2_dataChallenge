import os
import pandas as pd

class Dataimport: # class that imports two csv files (data and metadata)

    def __init__(self, datapath, metadatapath):  # paths must be .csv file
        if os.path.exists(datapath) and datapath[-3:]=='csv':
            self.datapath = datapath
            self.tidy = pd.read_csv(datapath, sep=",")
        else: raise Exception("This data-path does not exist or it is no .csv-file!")

        if os.path.exists(metadatapath) and metadatapath[-3:]=='csv':
            self.metadatapath = metadatapath
            self.tidyMet = pd.read_csv(metadatapath, sep=";", index_col=0)
        else: raise Exception("This metadata-path does not exist or it is no .csv-file!")
    
    def dataframe(self):
        tidyMer = pd.merge(self.tidy, self.tidyMet, how="left", on="R.FileName")
        return tidyMer[["R.FileName", "uniprot", "meanAbu", "Cancer"]]

# use as follows:                               data = Dataimport("rawData/tidy.csv","rawData/Metadata.csv")
# to reassign the dataframe produced, use:      df = data.dataframe()
# and you can have a look at the dataframe:     print(dataframe)