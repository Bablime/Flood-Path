import pandas as pd
df = pd.read_excel("unzipped_data/flood_shelter/reach_bgd_database_cyclone-shelters-in-ukhiya-and-teknaf_november_2019(Flood Shelter Location).xlsx", sheet_name='DRRO + other identified shelter')
print(df.columns.tolist())
df_roads = pd.read_csv("unzipped_data/local_road/Local Road DataSet.csv", nrows=1)
print(df_roads.columns.tolist())
