import pickle



with open('data/distance_finals_copy.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)


# the difference between the two distance matricies is minimal becuase there is often only one or two elements that are different
# so to make this more clear and produce more distinguishing results between the two distance matricies, we can square the distance matricies
# or we can amplify the distance between them by making them more different
first = distance_matrices[0]
second = distance_matrices[1]

print(first)
print(second)