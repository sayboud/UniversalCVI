import pandas as pd
import os

data_dir = os.path.dirname(__file__)

def load_dataset(name):
    path = os.path.join(data_dir, f"{name}.csv")
    return pd.read_csv(path)

R1_data = load_dataset("R1_data")
R2_data = load_dataset("R2_data")
R3_data = load_dataset("R3_data")
R4_data = load_dataset("R4_data")
R5_data = load_dataset("R5_data")
R6_data = load_dataset("R6_data")
R7_data = load_dataset("R7_data")
D1_data = load_dataset("D1_data")
D2_data = load_dataset("D2_data")
D3_data = load_dataset("D3_data")
D4_data = load_dataset("D4_data")
D5_data = load_dataset("D5_data")
D6_data = load_dataset("D6_data")
D7_data = load_dataset("D7_data")
D8_data = load_dataset("D8_data")
D9_data = load_dataset("D9_data")
D10_data = load_dataset("D10_data")