import os
import pandas as pd

class Dataimport: # class that imports two csv files (data and metadata)

    def __init__(self, datapath, metadatapath):  # paths must be .csv file
        if os.path.exists(datapath) and datapath[-3:]=='csv':
            self.datapath = datapath
        else: raise Exception("This data-path does not exist or it is no .csv-file!")

        if os.path.exists(metadatapath) and metadatapath[-3:]=='csv':
            self.metadatapath = metadatapath
        else: raise Exception("This metadata-path does not exist or it is no .csv-file!")
    
    def dataframe(self):
        datapath = self.datapath
        metadatapath = self.metadatapath
        tidy    = pd.read_csv(datapath, sep=",")
        tidyMet = pd.read_csv(metadatapath, sep=";", index_col=0)
        tidyMer = pd.merge(tidy, tidyMet, how="left", on="R.FileName")
        return tidyMer[["R.FileName", "uniprot", "meanAbu", "Cancer"]]


data = Dataimport("rawData/tidy.csv","rawData/Metadata.csv")
dataframe = data.dataframe()
print(dataframe)