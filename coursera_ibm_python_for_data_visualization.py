
# coding: utf-8

# Python Visualizations: Taken from: https://www.coursera.org/learn/python-for-data-visualization/

# In[1]:

get_ipython().magic('matplotlib inline')


# ### Backend Layer:

# #### Matplotlib Backend-Layer:  

# - <b>FigureCanvas</b>: matplotlib.backend_bases.FigureCanvas. the area onto which the figure is drawn.
# - <b>Renderer</b>: matplotlib.backend_bases.Renderer. knows how to plot on the canvas.
# - <b>Event</b>: matplotlib.backend_bases.Event. handles user-input as keystrokes and mouse clicks.

# #### Matplotlib Artist-Layer:  

# - <b>Artist</b>: knows how to use the Renderer to plot on the Canvas. Titles, Lines, ticks, images, etc.  
#     - <b>Primitive Artist</b>: Line2d, Rectangle, Circle, Text.
#     - <b>Composite Artist</b>: Axis, Ticks, <b>Axes</b>, <b>Figure</b>. May include other composite/promitive artists in it. For example, Figure may incluide Axes, Line2D, Text, etc.

# In[2]:

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure # Figure Artist

import numpy as np


# In[3]:

fig = Figure()
canvas = FigureCanvas(fig)


# In[4]:

x = np.random.randn(10000)


# In[5]:

ax = fig.add_subplot(111) # creates and Axes Artist. Matlab Convention


# In[6]:

ax.hist(x, 100)
ax.set_title("Normal Dist. $\mu=0, \sigma=1$")
fig.savefig("matplotlib_hist.png")


# ### Pyplot

# In[7]:

import matplotlib.pyplot as plt


# In[8]:

plt.hist(x, 100)
plt.title("Normal Dist. $\mu=0, \sigma=1$")
plt.savefig("matplotlib_hist.png")
plt.show()


# In[9]:

plt.plot(5,5,"o")
plt.show()


# In[10]:

plt.plot(5,5,"o")
plt.title("dot")
plt.xlabel("$x$")
plt.ylabel("$y$")


# ### Read Data

# In[11]:

import pandas as pd
from __future__ import print_function


# In[12]:

# df = pd.read_excel("https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx", 
#                    sheetname="Canada by Citizenship", 
#                    skiprows=range(20), 
#                    skip_footer=2)
# df.to_csv("data/immigrants_df.csv", index=False)


# In[13]:

df = pd.read_csv("data/immigrants_df.csv").set_index("OdName", drop=True)
df.index.name="Country"
df.head(2)


# ### Plotting with Pandas:

# In[14]:

# !pip install xlrd


# In[15]:

df[["1980", "2013"]].plot(kind="line")


# In[16]:

df["1980"].plot(kind="hist")


# ### Line Plots

# In[17]:

import matplotlib as mpl
import seaborn as sns


# In[18]:

years = list(map(str, range(1980, 2014)))
print(years)


# In[19]:

df.loc["Haiti", years].plot(kind="line")
plt.title("Immigrants Haiti")
plt.ylabel("# of Immigrants")
plt.xlabel("year")
plt.grid(which='minor', axis='both')
plt.show()


# ### Area Plots:

# In[20]:

df["Total"] = df[years].sum(axis=1)
df.head()


# In[21]:

df.sort_values(["Total"], ascending=False, axis=0, inplace=True)
df.head(2)


# In[22]:

df_top5 = df.head()[years].T
df_top5.head(3)


# In[23]:

df_top5.plot(kind="area", alpha=0.5)
plt.title("area plot")
plt.xlabel("$n$ immigrants")
plt.ylabel("Year")

plt.show()


# ### Histogram:

# In[24]:

df["2013"].plot(kind="hist")
plt.title("hist plot")
plt.xlabel("$n$ immigrants")
plt.ylabel("$n$ countries")

plt.show()


# In[25]:

# in order to align the bins edges with tick-marks:
count, bin_edges = np.histogram(df["2013"])

df["2013"].plot(kind="hist", xticks=bin_edges)
plt.title("hist plot")
plt.xlabel("$n$ immigrants")
plt.ylabel("$n$ countries")

plt.show()


# ### Bar-Charts:

# In[26]:

df_iceland = df.loc["Iceland", years]


# In[27]:

df_iceland.plot(kind="bar")
plt.title("bar plot")
plt.xlabel("Year")
plt.ylabel("$n$ immigrants")

plt.show()


# ### Pie Charts:

# In[28]:

df_continents = df.groupby("AreaName", axis=0).sum()


# In[29]:

df_continents[years]


# In[30]:

df_continents["Total"].plot(kind="pie")
plt.title("pie plot")

# equalize the aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.show()


# ### Box-Plot

# In[31]:

df_japan = df.loc[["Japan"], years].T
df_japan.head(3)


# In[32]:

df_japan.plot(kind="box")
plt.title("box plot")
plt.ylabel("$n$ immigrants")

plt.show()


# ### Scatter Plot

# In[33]:

# somehow, simply changing the col names makes scatter-plot not to recognize the columns. so this is an ugly hack
df_years = df[years].sum().to_frame()
df_years.reset_index(inplace=True, drop=False)
df_years.columns = ["year", "total"]
df_years.head(3)
df_years.to_csv("data/df_years.csv", index=False)
df_years = pd.read_csv("data/df_years.csv")


# In[34]:

df_years.plot(kind="scatter", x="year", y="total")
plt.title("scatter plot")

plt.show()


# ### Waffle Charts

# Mainly to display progress towards goals.  
# more contribution = more tiles

# In[35]:

df_3 = df.loc[["Denmark", "Norway", "Sweden"]]
df_3


# In[36]:

import matplotlib.pyplot as plt
from pywaffle import Waffle


# In[37]:

data = (df_3["Total"]//100).to_dict()
data


# In[38]:

data_vals = np.array(list(data.values()))
data_vals


# In[39]:

# option1:
colors = ("#232066", "#983D3D", "#DCB732")

# option3:
cmap = mpl.cm.Accent
mini, maxi = data_vals.min(), data_vals.max()
norm = mpl.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in data_vals]

# option2:
# from: https://matplotlib.org/examples/color/named_colors.html
colors = ("sienna", "grey", "cornflowerblue")

colors


# In[40]:

# available icons from: https://fontawesome.com/icons?from=io
icon = "apple"
icon = "child"
icon = "male"
icon = None


# In[41]:

legend_loc = {'loc': 'upper left', 'bbox_to_anchor': (1, 1)}


# In[42]:

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    colors=colors,
    legend=legend_loc,
    icons=icon, 
    icon_size=18, 
    icon_legend=True
)


# ### Word Clouds

# - a word's size according to its freq in the text
# - e.g., find most common words in text for labeling etc.

# In[43]:

from wordcloud import WordCloud
import re
import multidict
# examples of how to use it: https://github.com/amueller/word_cloud/tree/master/examples


# In[44]:

# get some text (Alice in Wonderland):
from urllib import request
url = "http://www.gutenberg.org/files/11/11-0.txt"
response = request.urlopen(url)
text = response.read().decode('utf8')
text = " ".join(text.split())


# In[45]:

wc = WordCloud(background_color="black", max_words=1000)


# In[46]:

wordcloud = wc.generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# ### Seaborn and Regression Plots:

# In[47]:

df_japan = df.loc[["Japan"], years].T
df_japan = df_japan.reset_index()
df_japan.index.name = "index"
df_japan.columns = ["Year", "Number"]

# same infantil bug - so we save to csv and re-read it:
df_japan.to_csv("data/df_japan.csv")
df_japan = pd.read_csv("data/df_japan.csv")


# In[48]:

ax = sns.regplot(x="Year", y="Number", data=df_japan, color='sienna', marker="o") # inclides 95% interval


# ### Folium - for geo data

# In[49]:

import folium # interactive maps...


# In[50]:

world_map = folium.Map()
world_map


# In[51]:

center_map = [56.130, -106.35] # for Canada
world_map = folium.Map(location=center_map, zoom_start=4)

world_map


# #### with tiles:

# In[52]:

center_map = [56.130, -106.35] # for Canada
# optional tiles here: https://github.com/python-visualization/folium/tree/master/folium/templates/tiles
# tiles = 'Stamen Toner'
# tiles = 'stamenterrain'
# tiles = 'openstreetmap'
# tiles="mapboxbright"
tiles="mapboxcontrolroom"
world_map = folium.Map(location=center_map, 
                       zoom_start=4,
                       tiles=tiles)

world_map


# #### add markers to a map

# In[64]:

center_map = [56.130, -106.35] # for Canada
tiles="mapboxcontrolroom"
canada_map = folium.Map(location=center_map, 
                       zoom_start=4,
                       tiles=tiles)


# In[65]:

ontario = folium.map.FeatureGroup()


# In[66]:

ontario_loc = [51.25, -85.32]
circle_radius = 8.
fill_color = "lightcoral"
color = "peachpuff"

ontario_circle = folium.CircleMarker(location=ontario_loc, 
                                     radius=circle_radius, 
                                     color=color, 
                                     fill_color=fill_color)


# In[67]:

folium.Marker(ontario_loc, popup="Ontario").add_to(canada_map)


# In[68]:

ontario.add_child(ontario_circle)


# In[69]:

canada_map.add_child(ontario)


# ### Choropleth Maps

# In[ ]:



