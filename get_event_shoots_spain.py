#!/usr/bin/env python
# coding: utf-8



import math
import pandas as pd
import numpy as np
import os

def calculate_shot_angle(x, y):
    #x = df['X']
    #y = df['Y']
    if x > 60:
        xp1, yp1, xp2, yp2 = 120, 36, 120, 44
    else:
        xp1, yp1, xp2, yp2 = 0, 36, 0, 44
    vec1 = (xp1 - x, yp1 - y)
    vec2 = (xp2 - x, yp2 - y)
    
    angle = np.arccos((vec1[0]*vec2[0] + vec1[1]*vec2[1])/(math.sqrt((vec1[0]**2 + vec1[1]**2)*(vec2[0]**2 + vec2[1]**2))))
    return angle

def calculate_shot_distance(x, y):
    if x > 60:
        px,py = 120, 40
    else:
        px,py = 0, 40
    
    
    distance = math.sqrt((px - x)**2 + (py - y)**2)
    return distance

spain_list = []
file_list = os.listdir('data/matches/11')
for i in file_list:
    spain_list.append(pd.read_json('data/matches/11/{}'.format(i)))

spain = pd.concat(spain_list)

event_list = []
for i in spain['match_id']:
    event_list.append(pd.read_json('data/events/{}.json'.format(str(i))))

df = pd.concat(event_list)




df = df[["minute", "type",
           "team", "player", "location",
           "shot"]]





df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)

df = df.drop(columns = "type")





df = df[df["type_id"] == 16]




df = df.drop(columns = ["type_id", "type_name"])




df = pd.concat([df, df["player"].apply(pd.Series).add_prefix('player_')], axis=1)
df = df.drop(columns = "player")




df = pd.concat([df, df["location"].apply(pd.Series).rename({0: 'X', 1: 'Y'}, axis=1)], axis=1)
df = df.drop(columns = "location")




df = pd.concat([df, df["team"].apply(pd.Series).add_prefix('team_')], axis=1)
df = df.drop(columns = "team")





df = pd.concat([df, df["shot"].apply(pd.Series)], axis=1)
df = df.drop(columns = "shot")





df = df.drop(columns = ["key_pass_id", "first_time", "aerial_won",
                   "deflected", "one_on_one", "open_goal",
                   "redirect", "saved_to_post", "saved_off_target",
                  "follows_dribble", "freeze_frame", "end_location"])




df = pd.concat([df, df["technique"].apply(pd.Series).add_prefix('technique_')], axis=1)
df = df.drop(columns = "technique")




df = pd.concat([df, df["outcome"].apply(pd.Series).add_prefix('outcome_')], axis=1)
df = df.drop(columns = "outcome")




df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)
df = df.drop(columns = "type")




df = pd.concat([df, df["body_part"].apply(pd.Series).add_prefix('body_part_')], axis=1)
df = df.drop(columns = "body_part")




df["Goal"] = df["outcome_name"] == "Goal"

df['theta'] = None
df['theta'] = df.apply(lambda row : calculate_shot_angle(row['X'],
                     row['Y']), axis = 1)
df = df.dropna(subset=['theta']) 
df['distance'] = None
df['distance'] = df.apply(lambda row : calculate_shot_distance(row['X'],
                     row['Y']), axis = 1)


df.to_csv("laliga_shots.csv")

