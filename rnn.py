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
    def __init__(self, hidden_dim, goals_size, embed_dim, vocabulary):
            super(RNNEncoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.embed_dim = embed_dim
            self.goals_size = goals_size
            self.vocabulary = vocabulary


            self.W_hx = nn.Linear(embed_dim + goals_size, hidden_dim)
            self.W_hh = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.embedding = nn.Embedding(len(vocabulary), embed_dim)
            

    def forward(self, input, prev_hidden, h_goals):

        #make sure input is N x input_dim, where N is batch size, which can just be 1
        h_goals = h_goals.view(1, -1)
        input = input.view(-1)

        assert(input.shape == torch.Size([1]))
        assert(h_goals.shape == torch.Size([1, self.goals_size]))
        assert(prev_hidden.shape == torch.Size([1, self.hidden_dim]))

        embedded_input = self.embedding(input)
        input_combined = torch.cat((embedded_input, h_goals), 1)
        new_hidden = self.W_hx(input_combined) + self.W_hh(prev_hidden)
        new_hidden = self.activation(new_hidden)
        return new_hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)

class RNNDecoder(nn.Module):
    def __init__(self, hidden_dim, goals_size, embed_dim, vocabulary):
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

            self.activation = nn.ReLU()
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, context, prev_hidden):
        #input is word embedding

        #make sure input is N x input_dim, where N is batch size, which can just be 1
        h_goals = h_goals.view(1, -1)
        input = input.view(-1)

        assert(input.shape == torch.Size([1]))
        assert(h_goals.shape == torch.Size([1, self.goals_size]))
        assert(prev_hidden.shape == torch.Size([1, self.hidden_dim]))

        input_combined = torch.cat((input, self.h_goals, context), 1)
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
    def __init__(self, goals, embed_dim, hidden_dim, vocabulary):
        super(GoalsEncoder, self).__init__()

        self.goals = goals
        self.embed_dim = embed_dim
        self.vocabulary = vocabulary


        self.inputLayer = nn.Linear(goals.shape[0], hidden_dim)
        self.outputLayer = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.inputLayer(input)
        output = self.activation(output)

        output = self.outputLayer(output)
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





def train(encoder, word, encoded_goals):

    encoder_optimizer.zero_grad()
    loss = 0



    #loss += criterion(...)


    loss.backward()
    encoder_optimizer.step()






def trainOverDataset2(trainExamples, vocab):



    vocab.append("<sos>")
    vocab.append("<eos>")
    vocab.append("YOU:")
    vocab.append("THEM:")
    vocab = sorted(vocab)

    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)

    encoder = RNNEncoder(hidden_dim=100, goals_size=6, embed_dim=20, vocabulary=vocab)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)

    decoder = RNNDecoder(hidden_dim=100, goals_size=6, embed_dim=20, vocabulary=vocab)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    ##### assuming context and target are both a list of tokens index
    for context, response_target in trainExamples:
        context_tensor = torch.tensor(context, dtype=torch.long)





def trainOverDataset(trainExamples, vocab):

    #import pdb; pdb.set_trace()

    encoder = RNNEncoder(hidden_dim=100, goals_size=6, embed_dim=20, vocabulary=vocab)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    vocab.append("<sos>")
    vocab.append("<eos>")
    vocab = sorted(vocab)

    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)

    inputTargetPairs = []


    for i, example in enumerate(trainExamples):
        print("example = ", exampleToString(example))
        agent_input = example["input"]
        dialogue = example["dialogue"]
        output = example["output"]
        partner_input = example["partner input"]

        encoded_goals = utils.toTensor(agent_input)

        running_context = []

        for sentence_i, sentence in enumerate(dialogue):
            
            if(sentence_i==0):
                running_context += sentence[1].split(" ")
                continue

            print("sentence = ", sentence[1])
            print("running context = ", running_context)

            speaker = sentence[0]
            utterance = sentence[1].split(" ")

            context = [wordToIndex["<sos>"]]
            prev_utterance = dialogue[sentence_i-1][1].split(" ")
            for word in prev_utterance:
                context.append(wordToIndex[word])
            context.append(wordToIndex["<eos>"])

            print("context = " , context)

            #target = [wordToIndex[""]]


            '''for word in utterance:
                train(encoder, encoder_optimizer, criterion, word, context, encoded_goals)'''

            running_context += sentence[1].split(" ")










if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    with open(args.train_vocab_json, "r") as fp:
        vocab = json.load(fp)

    #trainOverDataset(trainExamples, vocab)
    #trainOutputClassifier(trainExamples, vocab)

