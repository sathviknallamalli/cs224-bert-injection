#plot how the score metric (increase in probability dsitribution) changes 


import pandas as pd  
df = pd.read_csv("results.csv")  

#average the score metric each rank for each layer
result = df.groupby(['layer', 'rank']).agg({'score': 'mean'}).reset_index()
#make a plot where the x-axis is the layer and the y-axis is the score but we have different lines for each rank
#use matplot lib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
ax = sns.lineplot(x="layer", y="score", hue="rank", data=result)

#save the plot
fig = ax.get_figure()
fig.savefig("plot1.png")






#average the score metric each rank for each layer
result = df.groupby(['layer', 'theta']).agg({'score': 'mean'}).reset_index()

#plot this graph so that layer is on the x-axis, score is the y-axis, and each line is a different theta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
#make a line for each theta
ax = sns.lineplot(x="layer", y="score", hue="theta", data=result)
fig = ax.get_figure()
fig.savefig("plot2.png")