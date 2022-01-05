import os
from typing import List

import pandas as pd



"""
将[dataframe1, dataframe2, dataframe3] 这种结构进行存储 name为0.csv 1.csv 2.csv
"""
def saveDFListToFiles(spath: str, pds: List[pd.DataFrame]):
    if not os.path.exists(spath):
        os.makedirs(spath)
    for i in range(0, len(pds)):
        savefilepath = os.path.join(spath, str(i) + ".csv")
        pds[i].to_csv(savefilepath, index=False)
