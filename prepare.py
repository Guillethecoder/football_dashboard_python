import pandas as pd
import numpy as np
import os

#GET COMPETITONS
df = pd.read_json('data/competitions.json')
df = df[df["season_name"].isin(["2016/2017","2017/2018", "2018/2019", "2019/2020", "2020/2021"])]
df = df[df["competition_gender"] == "male"]

#JOIN MATCHES
matches_list = []

for j in df["competition_id"].unique():
    comp = df[df["competition_id"] == j]
    for i in comp["season_id"]:
        season = pd.read_json('data/matches/{}/{}.json'.format(j,i))
        matches_list.append(season)

matches = pd.concat(matches_list)
#print(matches[["competition_name", "season_name"]])

#JOIN EVENTS
event_list = []
for i in matches['match_id']:
    event = pd.read_json('data/events/{}.json'.format(str(i)))
    event["match_id"] = i
    event = event[["minute", "type",
           "team", "player", "location",
           "shot", "match_id"]]
    event_list.append(event)

df = pd.concat(event_list)
df = pd.merge(df, matches, on="match_id")

df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)
df = df.drop(columns = "type")
df = df[df["type_id"] == 16]
df = df.drop(columns = ["type_id", "type_name"])