import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

#read data
url = 'C:/Users/goop/Desktop/wind_data.txt'
data = pd.read_csv(url, sep =r'\s+', header = None, names = ['Year', 'Month', 'Day', 'Hour', 'WD', 'WS', 'U', 'V'])
checking = data['U'].isnull().values.any()
checking2 = data['V'].isnull().values.any()
data.WS = data.WS / 10.0 # *0.514444444
data.head()
from datetime import datetime
#data pre-processing
datetimes = []
year_v = data['Year'].values
month_v = data['Month'].values
day_v = data['Day'].values
hour_v = data['Hour'].values
U_v = data['U'].values
V_v = data['V'].values

for i in range(0, len(year_v)):
    datetimes.append(datetime(year_v[i], month_v[i], day_v[i], hour_v[i]))

data.index = pd.Series(datetimes)
data.index.name = 'Datetime'
print(data)
"""
# remove outliers (remove noise)
# 1) WD > 360 are outliers
# 2) WS > 10000 are outliers
outlier_index = np.logical_or(wind_data['WD'] > 360, wind_data['WS'] > 1000)
wind_data = wind_data.loc[~outlier_index, :]
print("{0} rows of data were removed".format(len(outlier_index)) )
"""

wind_2021 = data.loc[data['Year'] == 2021, ['Month', 'Day', 'Hour', 'WD', 'WS', 'U', 'V']]
#wd_2021 = data[['Month', 'Day', 'WD']]
#ws_2021 = data[['Month', 'Day', 'WS']]
wdir = wind_2021.loc[wind_2021['Day'] == 1, ['Day', 'WD']]
wspeed = wind_2021.loc[wind_2021['Day'] == 1, ['Day', 'WS']]

#function to calculate bearing
def dtoc(d):

    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

#Graph of wind direction
df = pd.DataFrame(wind_2021, columns = ['WD','WS'])

#convert the degree to bearing
for i in range(0, len(wdir)):
    df.iloc[i, df.columns.get_loc('WD')] = dtoc(df.iloc[i]['WD'])
##check the type of data
#print(type(df.iloc[0, df.columns.get_loc('WD')]))
print(df)
fig,ax = plt.subplots()
ax.set_ylabel('Wind direction')
ax.plot(df.loc['2021', 'WD'], marker='o', linestyle='-')
plt.show()
plt.clf()


#wind quiver modelling

plt.ion
fg = plt.figure()
ax = fg.add_subplot(111)

def uv_converter(wspd, wdir):
    rad = 4.0 * np.arctan(1) / 180.
    u = -wspd * np.sin(rad * wdir)
    v = -wspd * np.cos(rad * wdir)
    return u, v

if (checking or checking2):
    sdd, svv = uv_converter(wspeed.WS, wdir.WD)  # For wind forecasting without U, V
else:
    ud = pd.DataFrame(wind_2021, columns=['U'])
    vd = pd.DataFrame(wind_2021, columns=['V'])
    sdd = ud.squeeze()
    svv = vd.squeeze()

#print(type(u1))
#print(pdd)

for i in range(0, len(sdd)):
    plt.clf() # clearfy and prevent for duplication
    plt.quiver(-sdd[i], -svv[i], scale=25.0, color = 'red') # vector quiver
    plt.xlim([-5,5]) # range of x coor
    plt.ylim([-5,5]) # range of y coor
    # Show years month day and hour
    plt.text(x = 0.6, y = 0.0, s='{0}-{1}-{2}-{3}'.format(
            wspeed.index[i].year, wspeed.index[i].month, wspeed.index[i].day,
            wspeed.index[i].hour))
    plt.text(x = 0.6, y = -1.5, s='(U,V) = ({0},{1})'.format(round(sdd[i],2), round(svv[i],1)))
    plt.text(x = 0.6, y = -0.5, s='Wind direction: {0}'.format(round(wdir.WD[i], 1)))
    plt.text(x = 0.6, y = -1.0, s='Wind speed: {0} (m/s)'.format(round(wspeed.WS[i], 1)))
    plt.savefig('{0}.png'.format(i))
    fg.canvas.draw() #refresh
    fg.canvas.flush_events()

#Output of gif
import glob
import imageio
graphs = []
for i in range(0, len(ud)):
    # filename record all of the wind png from the dict
    filename = str('{0}.png'.format(i))
    # put all the png to ram and append them to the list
    graphs.append(imageio.imread(filename))

# use of mimsave function to form the gif
imageio.mimsave('C:/Users/2021Summer_R4b/Desktop/HKO/venv/wind.gif', graphs, duration=0.5, loop=1)
#imageio.mimsave(outputpath, graphs, duration=0.5, loop=1)



