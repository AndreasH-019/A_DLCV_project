import pandas as pd
import numpy as np
from scipy import stats
import os

# Liste over dine CSV-filer
root = "giraffe_dice"
csv_files = os.listdir(root)

for file in csv_files:
    # Indlæs CSV-filen
    df = pd.read_csv(os.path.join(root, file))

    # Hent data fra "mAP"-søjlen
    map_values = df["Value"]

    # Beregn gennemsnit og standardafvigelse
    mean_value = np.mean(map_values)
    std_dev = np.std(map_values, ddof=1)  # ddof=1 for at bruge stikprøvestandardafvigelse

    # Beregn 95% konfidensinterval
    confidence_interval = stats.t.interval(0.95, len(map_values)-1, loc=mean_value, scale=std_dev/np.sqrt(len(map_values)))
    width = (confidence_interval[1]-confidence_interval[0])
    # Udskriv resultater
    print(f"File: {file}")
    print(f"Gennemsnit: {mean_value}")
    print(f"95% Konfidensinterval: {confidence_interval}")
    print(f"width/2 = {width/2}\n")
    # print(f"gennemsnit 2: {confidence_interval[0]+width/2}\n")