import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FCPython
import os


file_list = os.listdir('data/matches/11')
spain = pd.read_json('data/matches/11/{}'.format(file_list[0]))
for i in file_list[1:]:
    spain.append(pd.read_json('data/matches/11/{}'.format(i)))


events = pd.read_json('data/events/{}.json'.format(spain['match_id'][0]))
for match in spain['match_id'][1:]:

    print(match)
    """df = pd.read_json('data/events/{}.json').format(match)
    events.append(df)"""

events = events[["minute", "type",
           "team", "player", "location",
           "shot"]]

df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)
df = df.drop(columns = "type")
df = df[df["type_id"] == 16]
df = df.drop(columns = ["type_id", "type_name"])

print(df.head())