import sys  
import os
import re
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from config import parse_arguments
from readData import exampleToString, parseExample
from rnn import OutputClassifier, GoalsEncoder
import utils




def findBestValidPrediction(outputs, input_goals):

    outputs = [outputs[i].data.numpy()[0] for i in range(len(outputs))]

    item_counts = [count  for count,value in input_goals]
    pairs = []
    for count1 in range(item_counts[0]+1):
        opponent_count1 = item_counts[0] - count1
        for count2 in range(item_counts[1]+1):
            opponent_count2 = item_counts[1] - count2
            for count3 in range(item_counts[2]+1):
                opponent_count3 = item_counts[2] - count3
                log_probability = outputs[0][count1]
                log_probability += outputs[1][count2]
                log_probability += outputs[2][count3]
                log_probability += outputs[3][opponent_count1]
                log_probability += outputs[4][opponent_count2]
                log_probability += outputs[5][opponent_count3]

                counts = [count1, count2, count3, opponent_count1, opponent_count2, opponent_count3]
                pairs.append((log_probability, counts))

    log_probability, prediction = max(pairs)
    return prediction




def eval_output(outputs, target, example, verbose = False):
        predictions = []
        prediction_probs = []
        for output_idx in range(6):
            max_val, max_idx = torch.max(outputs[output_idx], 1)
            predictions.append(max_idx.data.numpy()[0])
            prediction_probs.append(max_val.data.numpy()[0])

        target = [tensor.data.numpy()[0] for tensor in target]

        predictions2 = findBestValidPrediction(outputs, example["input"])

        num_correct = 0
        total_diff = 0

        for i in range(6):
            if (predictions2[i]==target[i]):
                num_correct += 1
            total_diff += int(abs(predictions2[i] - target[i]))

        if(verbose):
            print("example = ", example)
            print("target = ", target)
            #print("outputs = ", outputs)
            print("predictions = ", predictions)
            print("predictions2 = ", predictions2)
            #print("log probabilities = ", prediction_probs)

        return num_correct, total_diff

#train Examples is list of dictionary entries, loaded from json
def generateDataset(trainExamples, vocab, use_shuffle = True):
    
    vocab.append("<START>")
    vocab.append("<sos>")
    vocab.append("<eos>")
    vocab.append("YOU")
    vocab.append("THEM")
    if "<UNK>" not in vocab:
        vocab.append("<UNK>")
    vocab = sorted(vocab)

    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)

    contextTargetPairs = []

    for i, example in enumerate(trainExamples):
        #print("example = ", exampleToString(example))
        agent_input = example["input"]
        dialogue = example["dialogue"]
        output = example["output"]
        partner_input = example["partner input"]

        goals_tensor = utils.toTensor(agent_input, "goals")

        #for now, just use raw tensor, can also train an embedding
        encoded_goals = goals_tensor

        word_indexes = []

        for sentence in dialogue:
            speaker = sentence[0]
            utterance = [speaker]
            utterance += sentence[1].split(" ")
            utterance += ["<eos>"]

            utterance_idx = [wordToIndex[word] if word in wordToIndex else wordToIndex["<UNK>"] for word in utterance]
            word_indexes += utterance_idx

        words_tensor = torch.tensor(word_indexes, dtype=torch.long)
        #print("words tensor = ", words_tensor)

        
        output_vector = output[0] + output[1]
        if (-1 in output_vector):
            ##there was disagreement so don't bother training on this example
            continue
        output_tensors = []
        for output_idx in range(6):
            output_count = output_vector[output_idx]
            output_tensor = torch.tensor([output_count], dtype = torch.long)
            output_tensors.append(output_tensor)

        context = (words_tensor, encoded_goals)
        target = output_tensors

        contextTargetPairs.append((context, target, example))

    if (use_shuffle):
        random.shuffle(contextTargetPairs)

    return contextTargetPairs

def train(context, target, classifier, optimizer, criterion):

    optimizer.zero_grad()
    loss = 0

    words_tensor, encoded_goals = context

    num_words = words_tensor.size(0)

    hidden = classifier.initHidden()

    for i in range(num_words):
        outputs, hidden = classifier(words_tensor[i], hidden, encoded_goals)

    for output_idx in range(6):
        loss += criterion(outputs[output_idx], target[output_idx])

    loss.backward()
    optimizer.step()

    return loss.item()/num_words, outputs

def validate(validationSet, classifier, verbose = False, savedPath = None):

    if savedPath != None:
        classifier = torch.load("savedModels/outputClassifier.pth")

    classifier.eval()

    cummulative_num_correct = 0
    cummulative_total_diff = 0
    for context, target, example in validationSet:

        words_tensor, encoded_goals = context

        num_words = words_tensor.size(0)

        hidden = classifier.initHidden()

        for i in range(num_words):
            outputs, hidden = classifier(words_tensor[i], hidden, encoded_goals)

        num_correct, total_diff = eval_output(outputs, target, example, verbose=verbose)
        cummulative_num_correct +=num_correct
        cummulative_total_diff += total_diff

    return cummulative_num_correct, cummulative_total_diff



def trainOutputClassifier(trainExamples, valExamples, vocab):

    import pdb; pdb.set_trace()

    contextTargetPairs = generateDataset(trainExamples, vocab, use_shuffle=False)
    contextTargetPairs_val = generateDataset(valExamples, vocab, use_shuffle = True)

    #classifier = OutputClassifier(hidden_dim=256, goals_size=6, embed_dim=256, vocabulary = vocab)
    classifier = torch.load("savedModels/outputClassifier.pth")
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    criterion = nn.NLLLoss()

    #### training
    loss_plot = []
    iters_to_plot = 100
    iters_to_save = 500
    iters_to_validate = 200

    epoch_loss = 0
    num_correct_epoch = 0
    total_diff_epoch = 0
    num_correct_val_epoch = 0
    total_diff_val_epoch = 0
    epoch_loss_plot = []
    #num_correct_plot = []
    #total_diff_plot = []
    num_correct_val_plot = []
    total_diff_val_plot = []
    num_correct_train_plot = []
    total_diff_train_plot = []

    trainingSize = len(contextTargetPairs)
    validationSize = 500
    trainingSet = contextTargetPairs[:trainingSize]
    validationSet = contextTargetPairs_val[:validationSize]

    assert(len(trainingSet) == trainingSize)
    assert(len(validationSet) == validationSize)

    num_iters = len(contextTargetPairs) + 1
    for iter in range(num_iters):
        context, target, example = trainingSet[iter % trainingSize]

        print("iter = ", iter)

        if ((iter % trainingSize) == 0):
            epoch_loss_plot.append(epoch_loss/trainingSize)
            epoch_loss = 0

            #num_correct_plot.append(num_correct_epoch/trainingSize)
            #total_diff_plot.append(total_diff_epoch/trainingSize)
            #num_correct_epoch = 0
            #total_diff_epoch = 0


        loss, outputs = train(context, target, classifier, optimizer, criterion)

        num_correct, total_diff = eval_output(outputs, target, example)
        num_correct_epoch += num_correct
        total_diff_epoch += total_diff

        if ((iter % iters_to_validate) == 0):
            num_correct_val, total_diff_val = validate(validationSet, classifier)
            num_correct_val_plot.append(num_correct_val/validationSize)
            total_diff_val_plot.append(total_diff_val/validationSize)

            num_correct_train, total_diff_train = validate(trainingSet, classifier)
            num_correct_train_plot.append(num_correct_train/trainingSize)
            total_diff_train_plot.append(total_diff_train/trainingSize)


        loss_plot.append(loss)
        epoch_loss += loss


        if ((iter % iters_to_plot) == 0):
            with open("plots1/TrainingLossIter.json", "w") as fp:
                json.dump(loss_plot, fp)
            with open("plots1/TrainingLossEpoch.json", "w") as fp:
                json.dump(epoch_loss_plot, fp)
            with open("plots1/NumCorrectOutputsTrain.json", "w") as fp:
                json.dump(num_correct_train_plot, fp)
            with open("plots1/TotalDifferenceInOutputTrain.json", "w") as fp:
                json.dump(total_diff_train_plot, fp)
            with open("plots1/NumCorrectOutputsValidation.json", "w") as fp:
                json.dump(num_correct_val_plot, fp)
            with open("plots1/TotalDifferenceInOutputValidation.json", "w") as fp:
                json.dump(total_diff_val_plot, fp)

        if ((iter % iters_to_save) == 0):
            #torch.save(classifier.state_dict(), "savedModels/outputClassifier.pth")
            torch.save(classifier, "savedModels/outputClassifierNew.pth")


    torch.save(classifier, "savedModels/outputClassifierNew.pth")
    print("done")

    #savedModel = classifier.load_state_dict(torch.load("savedModels/outputClassifier.pth"))
    savedModel = torch.load("savedModels/outputClassifierNew.pth")


    ### evaluate:
    print("Validating on examples in training set:")
    validate(trainingSet[:10], classifier, verbose=True)

    print("Validating on unseen examples:")
    validate(validationSet, classifier, verbose=True)
    
    print("Checking validation with model loaded from file")
    validate(validationSet[:5], classifier, verbose=True, savedPath = "savedModels/outputClassifierNew.pth")



if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    with open(args.val_data_json, "r") as fp:
        valExamples = json.load(fp)

    with open(args.train_vocab_json, "r") as fp:
        vocab = json.load(fp)

    trainOutputClassifier(trainExamples, valExamples, vocab)










