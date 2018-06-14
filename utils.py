import sys  
import os
import re
import json

import numpy
import torch


def fileToList(filepath):
    if not os.path.isfile(filepath):
       print("File path {} does not exist. Exiting...".format(filepath))
       sys.exit()

    lines = [line.rstrip('\n') for line in open(filepath)]
    return lines


def getIndexTranslations(vocab):
    wordToIndex = {word:i for i, word in enumerate(vocab)}
    indexToWord = {i:word for i, word in enumerate(vocab)}

    return wordToIndex, indexToWord

def toTensor(input, input_type):
    if input_type == "goals":
        goals_size = 6
        num_items = 3
        output = torch.zeros(goals_size, dtype=torch.float)
        for i in range(num_items):
            output[2*i] = input[i][0]
            output[2*i +1] = input[i][1]
        return output
    elif input_type == "output":
        num_players = 2
        num_items = 3
        agentOutput = input[0]
        opponentOutput = input[1]
        output = torch.tensor(agentOutput + opponentOutput, dtype=torch.float)
        return output
    else:
        return None
