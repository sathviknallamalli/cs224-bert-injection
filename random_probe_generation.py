import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np


#SELECT RANK 32, 0.6 THETA, ALL LAYERS

for k in range(0, 12):
    for rank in [4, 8, 16, 32, 64]:
        b_matrix_path = "probes/r" + str(rank) + "/" + str(k) + "/predictor.params"
        import_b = torch.load(b_matrix_path, map_location='cpu')
        b_matrix = import_b['proj']

        #estimate mean and standard deviation of the probe matrix
        mean = torch.mean(b_matrix)
        std = torch.std(b_matrix)

        #randomly sample new values for each element of the matrix of a uniform distribution with the same mean and standard deviation
        new_b_matrix = torch.empty(b_matrix.size())
        for i in range(b_matrix.size()[0]):
            for j in range(b_matrix.size()[1]):
                new_b_matrix[i][j] = torch.normal(mean, std) 

        #save this new matrix
        torch.save({'proj': new_b_matrix}, "probes/r" + str(rank) + "/" + str(k) + "/predictor_random.params")