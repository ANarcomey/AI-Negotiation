import sys  
import os
import re
import json
import random
#import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from config import parse_arguments
from readData import exampleToString, parseExample
import utils


#nn.Linear: applies a linear transformation to the incoming data: y=Ax+b

class RNNEncoder(nn.Module):
    def __init__(self, hidden_dim, goals_size, embed_dim, vocabulary, useGRU = True):
            super(RNNEncoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.embed_dim = embed_dim
            self.goals_size = goals_size
            self.vocabulary = vocabulary


            self.W_hx = nn.Linear(embed_dim + goals_size, hidden_dim)
            self.W_hh = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.Tanh()
            self.embedding = nn.Embedding(len(vocabulary), embed_dim)

            self.GRU = nn.GRU(embed_dim + goals_size, hidden_dim)
            self.useGRU = useGRU
            

    def forward(self, input, prev_hidden, h_goals):

        #make sure input is N x input_dim, where N is batch size, which can just be 1
        h_goals = h_goals.view(1, -1)
        input = input.view(-1)

        assert(input.shape == torch.Size([1]))
        assert(h_goals.shape == torch.Size([1, self.goals_size]))
        assert(prev_hidden.shape == torch.Size([1, self.hidden_dim]))

        embedded_input = self.embedding(input)
        input_combined = torch.cat((embedded_input, h_goals), 1)

        if(self.useGRU):
            _, new_hidden = self.GRU(input_combined.view(1, 1, -1), prev_hidden.view(1, 1, -1))
            new_hidden = new_hidden.squeeze(0)
        else:
            new_hidden = self.W_hx(input_combined) + self.W_hh(prev_hidden)
            new_hidden = self.activation(new_hidden)

        return new_hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)

class RNNDecoder(nn.Module):
    def __init__(self, hidden_dim, goals_size, embed_dim, vocabulary, useGRU = True):
            super(RNNDecoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.goals_size = goals_size
            self.embed_dim = embed_dim
            self.vocabulary = vocabulary

            self.context_size = hidden_dim
            self.vocabulary_size = len(vocabulary)


            self.W_hy = nn.Linear(embed_dim + goals_size + self.context_size, hidden_dim)
            self.W_hh = nn.Linear(hidden_dim, hidden_dim)
            self.W_out = nn.Linear(hidden_dim, self.vocabulary_size)

            self.activation = nn.Tanh()
            self.embedding = nn.Embedding(len(vocabulary), embed_dim)
            self.softmax = nn.LogSoftmax(dim=1)

            self.GRU = nn.GRU(embed_dim + goals_size + self.context_size, hidden_dim)
            self.useGRU = useGRU

    def forward(self, input, context, prev_hidden, h_goals):
        #input is word embedding

        #make sure input is N x input_dim, where N is batch size, which can just be 1
        h_goals = h_goals.view(1, -1)
        input = input.view(-1)

        assert(input.shape == torch.Size([1]))
        assert(h_goals.shape == torch.Size([1, self.goals_size]))
        assert(prev_hidden.shape == torch.Size([1, self.hidden_dim]))
        assert(context.shape == torch.Size([1, self.context_size]))

        embedded_input = self.embedding(input)
        input_combined = torch.cat((embedded_input, h_goals, context), 1)

        if(self.useGRU):
            _, new_hidden = self.GRU(input_combined.view(1, 1, -1), prev_hidden.view(1, 1, -1))
            new_hidden = new_hidden.squeeze(0)
        else:
            new_hidden = self.W_hy(input_combined) + self.W_hh(prev_hidden)
            new_hidden = self.activation(new_hidden)

        output = self.W_out(new_hidden)
        output = self.softmax(output)
        #now we have vector where output[i] = probability that word i is the next word to output
        return output, new_hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)
        #maybe use context???

class GoalsEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(GoalsEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim


        self.inputLayer = nn.Linear(input_dim, hidden_dim)
        self.outputLayer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, input):
        input = input.view(1, -1)
        output = self.inputLayer(input)
        output = self.activation(output)

        output = self.outputLayer(output)
        output = self.activation(output)
        return output

class OutputClassifier(nn.Module):
    def __init__(self, hidden_dim, goals_size, embed_dim, vocabulary, useGRU = True):
        super(OutputClassifier, self).__init__()

        self.output_dim = 6
        self.max_item_count = 10


        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.goals_size = goals_size
        self.vocabulary = vocabulary


        self.W_hx = nn.Linear(embed_dim + goals_size, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = [nn.Linear(hidden_dim, self.max_item_count + 1) for i in range(self.output_dim)]

        #self.activation = nn.ReLU()
        self.activation = nn.Tanh()
        self.embedding = nn.Embedding(len(vocabulary), embed_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        self.GRU = nn.GRU(embed_dim + goals_size, hidden_dim)
        self.useGRU = useGRU

    def forward(self, input, prev_hidden, h_goals):

        #make sure input is N x input_dim, where N is batch size, which can just be 1
        h_goals = h_goals.view(1, -1)
        input = input.view(-1)

        assert(input.shape == torch.Size([1]))
        assert(h_goals.shape == torch.Size([1, self.goals_size]))
        assert(prev_hidden.shape == torch.Size([1, self.hidden_dim]))

        embedded_input = self.embedding(input)
        input_combined = torch.cat((embedded_input, h_goals), 1)
        if(self.useGRU):
            _, new_hidden = self.GRU(input_combined.view(1, 1, -1), prev_hidden.view(1, 1, -1))
            new_hidden = new_hidden.squeeze(0)
        else:
            new_hidden = self.W_hx(input_combined) + self.W_hh(prev_hidden)
            new_hidden = self.activation(new_hidden)
        

        outputs = [W_out_idx(new_hidden) for W_out_idx in self.W_out]
        outputs = [self.softmax(output_idx) for output_idx in outputs]
        return outputs, new_hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)




