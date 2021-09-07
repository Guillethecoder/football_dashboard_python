#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import os
import math
import pickle


# In[1]:


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


# In[16]:


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


# In[3]:


df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)
df = df.drop(columns = "type")
df = df[df["type_id"] == 16]
df = df.drop(columns = ["type_id", "type_name"])


# In[4]:


df


# In[5]:


df = pd.concat([df, df["player"].apply(pd.Series).add_prefix('player_')], axis=1)
df = df.drop(columns = "player")




df = pd.concat([df, df["location"].apply(pd.Series).rename({0: 'X', 1: 'Y'}, axis=1)], axis=1)
df = df.drop(columns = "location")




df = pd.concat([df, df["team"].apply(pd.Series).add_prefix('team_')], axis=1)
df = df.drop(columns = "team")





"""df = pd.concat([df, df["shot"].apply(pd.Series)], axis=1)
df = df.drop(columns = "shot")"""


# In[6]:


df = pd.concat([df, df["competition"].apply(pd.Series)], axis=1)
df = df.drop(columns = "competition")




df = pd.concat([df, df["season"].apply(pd.Series)], axis=1)
df = df.drop(columns = "season")




df = pd.concat([df, df["home_team"].apply(pd.Series)], axis=1)
df = df.drop(columns = "home_team")





df = pd.concat([df, df["away_team"].apply(pd.Series)], axis=1)
df = df.drop(columns = "away_team")


# In[7]:


df = pd.concat([df, df["competition_stage"].apply(pd.Series).add_prefix('competition_stage_')], axis=1)
df = df.drop(columns = ["competition_stage", "competition_stage_id"])


# In[8]:


df = pd.concat([df, df["stadium"].apply(pd.Series).add_prefix('stadium_')], axis=1)
df = df.drop(columns = ["stadium", "stadium_id", "stadium_0" , "stadium_country"])


# In[9]:


df = df.drop(columns=['match_status', 'match_status_360', 'last_updated',
       'last_updated_360', 'metadata', 'referee', 'home_team_gender',
                     'away_team_gender', 'home_team_group', 'away_team_group',
                     'managers', 'country', 'country_name'])


# In[10]:


df = pd.concat([df, df["shot"].apply(pd.Series)], axis=1)
df = df.drop(columns = "shot")


# In[11]:


df=df.drop(columns = ["key_pass_id", "first_time", "aerial_won",
                            'freeze_frame', 'one_on_one','saved_to_post',
                           'redirect','open_goal','redirect', 'deflected',
                           'saved_off_target', 'follows_dribble'])


# In[12]:


df = pd.concat([df, df["technique"].apply(pd.Series).add_prefix('technique_')], axis=1)
df = df.drop(columns = ["technique","technique_id"])


df = pd.concat([df, df["outcome"].apply(pd.Series).add_prefix('outcome_')], axis=1)
df = df.drop(columns = ["outcome","outcome_id"])


df = pd.concat([df, df["type"].apply(pd.Series).add_prefix('type_')], axis=1)
df = df.drop(columns = ["type","type_id"])


df = pd.concat([df, df["body_part"].apply(pd.Series).add_prefix('body_part_')], axis=1)
df = df.drop(columns = ["body_part","body_part_id"])


# In[13]:


df["Goal"] = df["outcome_name"] == "Goal"


# In[17]:


df['theta'] = None
df['theta'] = df.apply(lambda row : calculate_shot_angle(row['X'],
                     row['Y']), axis = 1)
df = df.dropna(subset=['theta']) 
df['distance'] = None
df['distance'] = df.apply(lambda row : calculate_shot_distance(row['X'],
                     row['Y']), axis = 1)


# In[20]:

filename = 'model_all_spanish_male.sav'
loaded_model = pickle.load(open(filename, 'rb'))

df['my_xg'] = loaded_model.predict_proba(df[['theta']])[:,1]
result = loaded_model.score(df[['theta']],df[['Goal']])
print(result)

df.to_csv("all_shots_16_20.csv",  index=False)

