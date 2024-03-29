
import regex as re

#iterate through each line
runnames = []
runscores = []
with open('nonrandom-combined.txt') as topo_file:
    for line in topo_file:
        if("NEW RUN:" in line):
            copy = line
            copy = copy.replace("NEW RUN: ", "")
            runnames.append(copy)
        if("Average increase, decrease scores" in line):
            #replace string 
            copy = line.replace("Average increase, decrease scores is ", "")
            copy = copy.replace(".\n", "")
            
            val = float(copy)
            runscores.append(val)

print(len(runnames))

import csv

csv_file_path = "fixed-nonrandom.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)    
    writer.writerow(["layer", "rank", "theta", "score"])
    for i in range(len(runnames)):
        name = runnames[i]
        numbers = re.findall(r'\d+\.*\d*', name)
        numbers.append(runscores[i])
        
        writer.writerow(numbers)


