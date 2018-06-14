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
from rnn import OutputClassifier, GoalsEncoder, RNNEncoder, RNNDecoder
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
            print("target = ", target)
            #print("outputs = ", outputs)
            print("predictions = ", predictions)
            print("predictions2 = ", predictions2)
            #print("log probabilities = ", prediction_probs)

        return num_correct, total_diff

#train Examples is list of dictionary entries, loaded from json
def generateDataset(trainExamples, vocab, use_shuffle = True, skip_disagreement = False):
    
    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)

    conversations = []

    for i, example in enumerate(trainExamples):
        #print("example = ", exampleToString(example))
        agent_input = example["input"]
        dialogue = example["dialogue"]
        output = example["output"]
        partner_input = example["partner input"]

        if skip_disagreement:
            output_vector = output[0] + output[1]
            if (-1 in output_vector):
                ##there was disagreement so don't bother training on this example
                continue

        goals_tensor = utils.toTensor(agent_input, "goals")

        word_indexes = []

        running_context = [wordToIndex["<START>"]]
        contextResponsePairs = []

        for sentence_i, sentence in enumerate(dialogue):
    
            speaker = sentence[0]
            utterance = [speaker]
            utterance += sentence[1].split(" ")
            utterance += ["<eos>"]

            utterance_idx = [wordToIndex[word] if word in wordToIndex else wordToIndex["<UNK>"] for word in utterance]

            #Clip the speaker token or eos token???
            target_tensor = torch.tensor(utterance_idx, dtype=torch.long)

            running_context_tensor = torch.tensor(running_context, dtype=torch.long)
            contextResponsePairs.append((running_context_tensor, target_tensor))

        
            running_context += utterance_idx


        conversations.append((goals_tensor, contextResponsePairs, example))
        

    if (use_shuffle):
        random.shuffle(conversations)

    return conversations

def train(goals_tensor, contextResponsePairs, goals_encoder, encoder, decoder, \
    goals_encoder_optimizer, encoder_optimizer, decoder_optimizer, criterion, indexToWord, wordToIndex):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    goals_encoder_optimizer.zero_grad()
    loss = 0

    encoder.train()
    decoder.train()
    goals_encoder.train()

    total_num_tokens = 0
    for context, response in contextResponsePairs:

        #use these for debugging if you want to see the context and response in words:
        #context_words = [indexToWord[int(index.numpy())] for index in context]
        #response_words = [indexToWord[int(index.numpy())] for index in response]


        #encode:
        encoded_goals = goals_encoder(goals_tensor)


        hidden = encoder.initHidden()
        num_context_words = context.size(0)
        for i in range(num_context_words):
            hidden = encoder(context[i], hidden, encoded_goals)

        #decode
        num_response_words = response.size(0)
        encoded_context = hidden
        hidden = decoder.initHidden()
        decoder_input = torch.tensor([wordToIndex["<sos>"]], dtype=torch.long)[0]
        for i in range(num_response_words):
            output, hidden = decoder(decoder_input, encoded_context, hidden, encoded_goals)

            loss += criterion(output, response[i].view(-1))
            total_num_tokens += 1

            # artifically choose the actual correct response token as input to the next step
            decoder_input = response[i]


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    goals_encoder_optimizer.step()

    return loss.item()/total_num_tokens

def validate(validationSet, goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex, verbose = False, savedEncoderPath = None, savedDecoderPath = None):

    if savedEncoderPath != None:
        encoder = torch.load(savedEncoderPath)

    if savedDecoderPath != None:
        decoder = torch.load(savedDecoderPath)

    encoder.eval()
    decoder.eval()
    goals_encoder.eval()

    num_tokens_correct = 0
    total_tokens = 0
    loss = 0

    for goals_tensor, contextResponsePairs, example in validationSet:

        encoded_goals = goals_encoder(goals_tensor)

        for context, response in contextResponsePairs:

            #use these for debugging if you want to see the context and response in words:
            context_words = [indexToWord[int(index.numpy())] for index in context]
            response_words = [indexToWord[int(index.numpy())] for index in response]

            if(verbose):
                print("context words = ", context_words)
                print("response words = ", response_words)


            #encode:
            hidden = encoder.initHidden()
            num_context_words = context.size(0)
            for i in range(num_context_words):
                hidden = encoder(context[i], hidden, encoded_goals)

            #decode
            selected_tokens = []

            num_response_words = response.size(0)
            encoded_context = hidden
            hidden = decoder.initHidden()
            decoder_input = torch.tensor([wordToIndex["<sos>"]], dtype=torch.long)[0]
            for i in range(0,num_response_words):
                output, hidden = decoder(decoder_input, encoded_context, hidden, encoded_goals)

                loss += criterion(output, response[i].view(-1))

                #categorical = torch.distributions.Categorical(output)
                #selected_token = categorical.sample()
                selected_token_probability, selected_token = torch.max(output, 1)

                selected_token_as_word = indexToWord[int(selected_token[0].numpy())]
                selected_tokens.append(selected_token_as_word)

                num_tokens_correct += 1 if (selected_token[0] == response[i]) else 0
                total_tokens += 1
                #print("output token = ", indexToWord[int(selected_token.numpy())])

                # artifically choose the actual correct response token as input to the next step
                decoder_input = response[i]

            if(verbose):
                print("Selected tokens = ", selected_tokens)
    
    if(verbose):
        print("Total Tokens = ", total_tokens)
        print("Number of correct tokens = ", num_tokens_correct)
    return num_tokens_correct/(total_tokens), loss.item()/total_tokens



def trainGenerativeModel(trainExamples, valExamples, vocab):

    import pdb; pdb.set_trace()

    vocab.append("<START>")
    vocab.append("<sos>")
    vocab.append("<eos>")
    vocab.append("YOU")
    vocab.append("THEM")
    if "<UNK>" not in vocab:
        vocab.append("<UNK>")
    vocab = sorted(vocab)

    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)


    contextTargetPairs = generateDataset(trainExamples, vocab, use_shuffle=False)
    contextTargetPairs_val = generateDataset(valExamples, vocab, use_shuffle = True)

    

    '''encoder = RNNEncoder(hidden_dim=128, goals_size=64, embed_dim=256, vocabulary = vocab)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)

    decoder = RNNDecoder(hidden_dim=128, goals_size=64, embed_dim=256, vocabulary = vocab)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    goals_encoder = GoalsEncoder(hidden_dim=64, output_dim=64, input_dim=6)
    goals_encoder_optimizer = optim.SGD(goals_encoder.parameters(), lr=0.01)'''
    encoder = torch.load("savedModels/encoder.pth")
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)

    decoder = torch.load("savedModels/decoder.pth")
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    goals_encoder = GoalsEncoder(hidden_dim=64, output_dim=64, input_dim=6)
    goals_encoder_optimizer = optim.SGD(goals_encoder.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    #### training
    loss_plot = []
    iters_to_plot = 100
    iters_to_save = 250
    iters_to_validate = 200

    epoch_loss = 0
    epoch_loss_plot = []
    token_accuracy_train_plot = []
    token_accuracy_val_plot = []

    token_num_correct_epoch = 0
    token_total_epoch = 0
    token_accuracy_epoch_plot = []
    loss_val_plot = []
    '''num_correct_epoch = 0
    total_diff_epoch = 0
    num_correct_val_epoch = 0
    total_diff_val_epoch = 0
    
    num_correct_plot = []
    total_diff_plot = []
    num_correct_val_plot = []
    total_diff_val_plot = []'''

    trainingSize = 1000
    validationSize = 100
    trainingSet = contextTargetPairs[:trainingSize]
    validationSet = contextTargetPairs_val[:validationSize]

    assert(len(trainingSet) == trainingSize)
    assert(len(validationSet) == validationSize)

    num_iters = 2001
    for iter in range(num_iters):
        goals_tensor, contextResponsePairs, example = trainingSet[iter % trainingSize]

        print("iter = ", iter)

        if ((iter % trainingSize) == 0):
            epoch_loss_plot.append(epoch_loss/trainingSize)
            epoch_loss = 0

            '''token_accuracy_epoch_plot.append(token_num_correct_epoch/token_total_epoch)
            token_num_correct_epoch = 0
            token_total_epoch = 0'''
            '''num_correct_plot.append(num_correct_epoch/trainingSize)
            total_diff_plot.append(total_diff_epoch/trainingSize)
            num_correct_epoch = 0
            total_diff_epoch = 0'''


        loss = train(goals_tensor, contextResponsePairs, goals_encoder, encoder, decoder, \
                goals_encoder_optimizer, encoder_optimizer, decoder_optimizer, criterion, indexToWord, wordToIndex)


        if ((iter % iters_to_validate) == 0):

            #print("Validating on training set:")
            accuracy_train, _ = validate(trainingSet, goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex)
            token_accuracy_train_plot.append(accuracy_train)

            #print("Validating on validation set:")
            accuracy_val, loss_val = validate(validationSet, goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex)
            token_accuracy_val_plot.append(accuracy_val)
            loss_val_plot.append(loss_val)

        loss_plot.append(loss)
        epoch_loss += loss


        if ((iter % iters_to_plot) == 0):
            with open("plots/TrainingLossIter.json", "w") as fp:
                json.dump(loss_plot, fp)
            with open("plots/TrainingLossEpoch.json", "w") as fp:
                json.dump(epoch_loss_plot, fp)
            with open("plots/TrainingAccuracy.json", "w") as fp:
                json.dump(token_accuracy_train_plot, fp)
            with open("plots/ValidationAccuracy.json", "w") as fp:
                json.dump(token_accuracy_val_plot, fp)
            with open("plots/ValidationLoss.json", "w") as fp:
                json.dump(loss_val_plot, fp)
        
        if ((iter % iters_to_save) == 0):
            torch.save(encoder, "savedModels/encoder.pth")
            torch.save(decoder, "savedModels/decoder.pth")


    torch.save(encoder, "savedModels/encoder.pth")
    torch.save(decoder, "savedModels/decoder.pth")
    print("\n\n\nFinished Training\n\n\n")

    ### evaluate:
    print("Validating on examples in training set:")
    validate(trainingSet[:10], goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex, verbose=True)

    print("Validating on unseen examples:")
    validate(validationSet, goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex, verbose=True)
    
    print("Checking validation with model loaded from file")
    validate(validationSet[:5], goals_encoder, encoder, decoder, criterion, indexToWord, wordToIndex, verbose=True, \
        savedEncoderPath = "savedModels/encoder.pth", savedDecoderPath = "savedModels/decoder.pth")



if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    with open(args.val_data_json, "r") as fp:
        valExamples = json.load(fp)

    with open(args.train_vocab_json, "r") as fp:
        vocab = json.load(fp)

    trainGenerativeModel(trainExamples, valExamples, vocab)










