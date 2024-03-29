import csv
import pandas as pd

data2 = pd.read_csv('results.csv')
#iterate through each row of csv
length = len(data2)



# counter = 0
# for index, row in data2.iterrows():
#     #print with '&' seperating each row element
#     if counter < length / 2:
#         print(str(int (row['layer'])) + " & " + str(int (row['rank'])) + " & " + str((row['theta'])) + " & "+ str(row['score'].round(3)) + "\\\\")
#     counter += 1
    

# data = pd.read_csv('fixed-nonrandom.csv')
# data3 = pd.read_csv('fixed-random.csv')
# theta3 = list(data[data['theta'] == 0.3]['score'])
# theta6 = list(data[data['theta'] == 0.6]['score'])
# theta1 = list(data[data['theta'] == 1.0]['score'])

# thetavals = [0.3, 0.6, 1.0]
# maxtheta = [max(theta3), max(theta6), max(theta1)]

# #get the same data for data3 and plot it on the same graph, as two colors
# theta3 = list(data3[data3['theta'] == 0.3]['score'])
# theta6 = list(data3[data3['theta'] == 0.6]['score'])
# theta1 = list(data3[data3['theta'] == 1.0]['score'])

# import matplotlib.pyplot as plt

# plt.scatter(thetavals, maxtheta, color = 'blue')
# plt.scatter(thetavals, [max(theta3), max(theta6), max(theta1)], color = 'red')
# plt.xlabel('Theta')
# plt.ylabel('Score')
# plt.title('Maximum Score based on Theta value - Test')
# #add the gridlines to the plot
# #add legend
# plt.legend(['B-probe', 'Random probe'])
# plt.grid(True)
# plt.savefig('max-theta-score-test.pdf', format='pdf')



# rank4 = list(data[data['rank'] == 4]['score'])
# rank8 = list(data[data['rank'] == 8]['score'])
# rank16 = list(data[data['rank'] == 16]['score'])
# rank32 = list(data[data['rank'] == 32]['score'])
# rank64 = list(data[data['rank'] == 64]['score'])

# rankvals = [4, 8, 16, 32, 64]
# maxrank = [max(rank4), max(rank8), max(rank16), max(rank32), max(rank64)]

# #get the same data for data3 and plot it on the same graph, as two colors
# rank4 = list(data3[data3['rank'] == 4]['score'])
# rank8 = list(data3[data3['rank'] == 8]['score'])
# rank16 = list(data3[data3['rank'] == 16]['score'])
# rank32 = list(data3[data3['rank'] == 32]['score'])
# rank64 = list(data3[data3['rank'] == 64]['score'])



# plt.clf()
# #make a scatter plot of the max scores for each rank value
# plt.scatter(rankvals, maxrank, color = 'blue')
# plt.scatter(rankvals, [max(rank4), max(rank8), max(rank16), max(rank32), max(rank64)], color = 'red')
# plt.xlabel('Rank')
# plt.ylabel('Score')
# plt.title('Maximum Score based on Probe Rank - Test')
# #add the gridlines to the plot
# plt.grid(True)
# #add legend
# plt.legend(['B-probe', 'Random probe'])
# plt.savefig('max-rank-score-test.pdf', format='pdf')

# data = pd.read_csv('results.csv')
# rank4 = list(data[data['rank'] == 4]['score'])
# rank8 = list(data[data['rank'] == 8]['score'])
# rank16 = list(data[data['rank'] == 16]['score'])
# rank32 = list(data[data['rank'] == 32]['score'])
# rank64 = list(data[data['rank'] == 64]['score'])

# rankvals = [4, 8, 16, 32, 64]
# maxrank = [max(rank4), max(rank8), max(rank16), max(rank32), max(rank64)]

# import matplotlib.pyplot as plt
# #make a scatter plot of the max scores for each rank value
# plt.scatter(rankvals, maxrank, color = 'blue')
# plt.xlabel('Rank')
# plt.ylabel('Score')
# plt.title('Maximum Score based on Probe Rank - Train')
# #add the gridlines to the plot
# plt.grid(True)
# plt.savefig('max-rank-score-train.pdf', format='pdf')

# plt.clf()

# theta01 = list(data[data['theta'] == 0.01]['score'])
# theta15 = list(data[data['theta'] == 0.15]['score'])
# theta2 = list(data[data['theta'] == 0.2]['score'])
# theta25 = list(data[data['theta'] == 0.25]['score'])
# theta3 = list(data[data['theta'] == 0.3]['score'])
# theta6 = list(data[data['theta'] == 0.6]['score'])
# theta1 = list(data[data['theta'] == 1.0]['score'])

# thetavals = [0.01, 0.15, 0.2, 0.25, 0.3, 0.6, 1.0]
# maxtheta = [max(theta01), max(theta15), max(theta2), max(theta25), max(theta3), max(theta6), max(theta1)]

# #make a scatter plot of the max scores for each theta value
# plt.scatter(thetavals, maxtheta, color = 'blue')
# plt.xlabel('Theta')
# plt.ylabel('Score')
# plt.title('Maximum Score based on Theta Value - Train')
# #add the gridlines to the plot
# plt.grid(True)
# plt.savefig('max-theta-score-train.pdf', format='pdf')

data = pd.read_csv('results.csv')
layer0 = list(data[data['layer'] == 0]['score'])
layer1 = list(data[data['layer'] == 1]['score'])
layer2 = list(data[data['layer'] == 2]['score'])
layer3 = list(data[data['layer'] == 3]['score'])
layer4 = list(data[data['layer'] == 4]['score'])
layer5 = list(data[data['layer'] == 5]['score'])
layer6 = list(data[data['layer'] == 6]['score'])
layer7 = list(data[data['layer'] == 7]['score'])
layer8 = list(data[data['layer'] == 8]['score'])
layer9 = list(data[data['layer'] == 9]['score'])
layer10 = list(data[data['layer'] == 10]['score'])
layer11 = list(data[data['layer'] == 11]['score'])

layervals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
maxlayer = [max(layer0), max(layer1), max(layer2), max(layer3), max(layer4), max(layer5), max(layer6), max(layer7), max(layer8), max(layer9), max(layer10), max(layer11)]

import matplotlib.pyplot as plt
#make a scatter plot of the max scores for each layer value
plt.scatter(layervals, maxlayer, color = 'blue')
plt.xlabel('Layer')
plt.ylabel('Score')
plt.title('Maximum Score based on Layer of Injection - Train')
#add the gridlines to the plot
plt.grid(True)
#add line connecting the points 
plt.plot(layervals, maxlayer, color = 'blue')
plt.savefig('max-layer-score-train.pdf', format='pdf')