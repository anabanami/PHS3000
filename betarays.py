import monashspa.PHS3000 as spa
import pandas as pd
import pytz
data = spa.betaray.read_data(r'betaray_data.csv')

valid_data = []
i = 0
for run in data:
    print(f'{i=}')
    if run[0] == pd.Timestamp('2020-08-06 13:39:36+10:00', tz=pytz.FixedOffset(600)):
        valid_data.append(data[i:])
        continue
    i+=1





print(f'{valid_data=}')
