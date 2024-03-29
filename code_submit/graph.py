import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('fixed-random.csv')
# Data for first dataset


# Data for second dataset (dummy data)
data2 = pd.read_csv('fixed-nonrandom.csv')


# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

#make the subplots vertically stacked instead of side by side
fig, axs = plt.subplots(2, 1, figsize=(12, 8))



# Create subplots
#fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot for first dataset
axs[0].set_title('random probe')
for theta in df1['theta'].unique():
    subset = df1[df1['theta'] == theta]
    axs[0].plot(subset['rank'], subset['score'], marker='o', label=f'Theta={theta}')
axs[0].set_xlabel('Rank')
axs[0].set_ylabel('Score')  
axs[0].set_xticks([4, 8, 16, 32, 64]) 
axs[0].legend()
axs[0].grid(True)

# Plot for second dataset
axs[1].set_title('B probe')
for theta in df2['theta'].unique():
    subset = df2[df2['theta'] == theta]
    axs[1].plot(subset['rank'], subset['score'], marker='o', label=f'Theta={theta}')
axs[1].set_xlabel('Rank')
axs[1].set_ylabel('Score')
axs[1].set_xticks([4, 8, 16, 32, 64]) 
axs[1].legend()
axs[1].grid(True)

#make both plots have the same y-axis
axs[0].set_ylim(-.02, .08)
axs[1].set_ylim(-.02, .08)

plt.tight_layout()
plt.savefig('randomvsnonrandom.pdf', format='pdf')
#plt.show()