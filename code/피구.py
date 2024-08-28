import pandas as pd
import numpy as np

df = pd.read_clipboard()
df = df.sort_values("성별", ignore_index=True)

df_f = df.loc[0:12]
df_m = df.loc[13:]

np.random.seed(20240827)
team1_f = np.random.choice(13, 6, replace=False)
team1_m = np.random.choice(11, 6, replace=False)

team1 = pd.concat([df_f.iloc[team1_f,],
                   df_m.iloc[team1_m,]])
team1

team2 = df.drop(team1.index)
team2

pd.concat([team1.reset_index(drop=True), team2.reset_index(drop=True)], axis=1)