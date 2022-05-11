import os
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

directory_path = '/home/student/data/train/'
directory = os.fsencode(directory_path)
count = 1
data_dict = dict()

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    full_file_path = directory_path + filename
    name, end = os.path.splitext(os.path.basename(filename))
    _, patient_id = name.split('_')
    patient_df = pd.read_csv(full_file_path, delimiter='|')
    sepsis_label = patient_df['SepsisLabel'].tolist()
    if 1 in sepsis_label:  # slicing input data
        ind = sepsis_label.index(1)
        patient_df = patient_df[:ind+1]
    data_dict[int(patient_id)] = patient_df


data_dict = OrderedDict(sorted(data_dict.items()))
print(data_dict)
