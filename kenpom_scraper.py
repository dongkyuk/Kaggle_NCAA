# Source: https://www.kaggle.com/walterhan/scrape-kenpom-data
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re

# Create a method that parses a given year and spits out a raw dataframe
def import_raw_year(filename, year):
    delimiter='\t+'
    df = pd.read_csv(filename, delimiter=delimiter, names = 
                ["rank", "TeamName", "conference", "record", "adj_em", "adj_o", 
                "adj_o_rank", "adj_d", "adj_d_rank", "adj_tempo", "adj_tempo_rank", 
                "luck", "luck_rank", "sos_adj_em", "sos_adj_em_rank", "sos_adj_o",
                "sos_adj_o_rank","sos_adj_d", "sos_adj_d_rank", "nc_sos_adj_em", 
                "nc_sos_adj_em_rank", "Season"])
    df['Season'] = year
    return df


# # Import all the years into a singular dataframe
df = pd.DataFrame()
for year in range(2011, 2020): 
    df = pd.concat((df, import_raw_year(f"data/kenpom/kenpom_{year}.txt", year)), axis=0)
print(df)



df['TeamName'] = df['TeamName'].astype('string')

#### Clean Team Names so that they can be merged to NCAA data
# Replacing Southern with Southen Univ forces recorrecting TX Southern & Miss Southern
df.replace("Cal St.","CS", inplace = True)
df.replace("Albany","SUNY Albany", inplace = True)
df.replace("Abilene Christian","Abilene Chr", inplace = True)
df.replace("American","American Univ", inplace = True)
df.replace("Arkansas Little Rock","Ark Little Rock", inplace = True)
df.replace("Arkansas Pine Bluff","Ark Pine Bluff", inplace = True)
df.replace("Boston University","Boston Univ", inplace = True)
df.replace("Central Michigan","C Michigan", inplace = True)
df.replace("Central Connecticut","Central Conn", inplace = True)
df.replace("Coastal Carolina","Coastal Car", inplace = True)
df.replace("East Carolina","E Kentucky", inplace = True)
df.replace("Eastern Washington","E Washington", inplace = True)
df.replace("East Tennessee St.","ETSU", inplace = True)
df.replace("Fairleigh Dickinson","F Dickinson", inplace = True)
df.replace("Florida Atlantic","FL Atlantic", inplace = True)
df.replace("Florida Gulf Coast","FL Gulf Coast", inplace = True)
df.replace("George Washington","G Washington", inplace = True)
df.replace("Illinois Chicago","IL Chicago", inplace = True)
df.replace("Kent St.","Kent", inplace = True)
df.replace("Monmouth","Monmouth NJ", inplace = True)
df.replace("Mississippi Valley St.","MS Valley St", inplace = True)
df.replace("Mount St Mary's","Mt St Mary's", inplace = True)
df.replace("Montana St.","MTSU", inplace = True)
df.replace("Northern Colorado","N Colorado", inplace = True)
df.replace("North Dakota St.","N Dakota St", inplace = True)
df.replace("Northern Kentucky","N Kentucky", inplace = True)
df.replace("North Carolina A&T","NC A&T", inplace = True)
df.replace("North Carolina Central","NC Central", inplace = True)
df.replace("North Carolina St.","NC State", inplace = True)
df.replace("Northwestern St.","Northwestern LA", inplace = True)
df.replace("Prairie View A&M","Prairie View", inplace = True)
df.replace("South Carolina St.","S Carolina St", inplace = True)
df.replace("South Dakota St.","S Dakota St", inplace = True)
df.replace("Southern Illinois","S Illinois", inplace = True)
df.replace("Southeastern Louisiana","SE Louisiana", inplace = True)
df.replace("Stephen F Austin","SF Austin", inplace = True)
df.replace("Southern","Southern Univ", inplace = True)
df.replace("Southern Univ Miss","Southern Miss", inplace = True)
df.replace("Saint Joseph's","St Joseph's PA", inplace = True)
df.replace("Saint Louis","St Louis", inplace = True)
df.replace("Saint Mary's","St Mary's CA", inplace = True)
df.replace("Saint Peter's","St Peter's", inplace = True)
df.replace("Texas A&M Corpus Chris","TAM C. Christi", inplace = True)
df.replace("Troy St.","Troy", inplace = True)
df.replace("Texas Southern Univ","TX Southern", inplace = True)
df.replace("Louisiana Lafayette","Louisiana", inplace = True)
df.replace("UTSA","UT San Antonio", inplace = True)
df.replace("Western Michigan","W Michigan", inplace = True)
df.replace("Green Bay","WI Green Bay", inplace = True)
df.replace("Milwaukee","WI Milwaukee", inplace = True)
df.replace("Western Kentucky","WKU", inplace = True)
df.replace("College of Charleston","Col Charleston", inplace = True)
df.replace("Loyola Chicago","Loyola-Chicago", inplace = True)
def x(str_in):
    return str_in.replace(".", "")
df['TeamName'] = df['TeamName'].apply(x)

print(df)
print(df.info())

teams = pd.read_csv('data/MTeams.csv')
df = df.merge(teams, on =['TeamName'], how='left')
print(df)

df.to_csv('Mkenpom2021.csv', index=False)