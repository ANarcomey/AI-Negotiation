import sys  
import os
import re
import json
import random

import numpy as np
import torch

from config import parse_arguments
from readData import exampleToString, parseExample
from rnn import OutputClassifier, GoalsEncoder, RNNEncoder, RNNDecoder
from trainOutputClassification import findBestValidPrediction
import utils

items = ["book", "hat", "ball"]
plural_items = ["books", "hats", "balls"]


word_to_number = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
                    "both":2}


def countsToString(counts):
    result = "Item Counts: "
    for i, item in enumerate(items):
        result += "{}={}, ".format(item, counts[i])
    #trim last comma and space
    return result[:-2]

def tokenize(input_str):

    #surround punctuation marks with spaces
    input_str = re.sub("([?,.!;])", r' \1 ', input_str)
    input_str = re.sub('\s{2,}', ' ', input_str)

    words = input_str.split(" ")
    return words

def extractItemCounts(input_str):


    words = tokenize(input_str)

    estimated_counts = [0]*len(items)
    words_lower = [word.lower() for word in words]
    for i, item in enumerate(items):
        if (item in words_lower):
            index = words_lower.index(item)
            if(index==0):
                #assume one item implied
                estimated_counts[i] = 1
                continue

            prev_word_lower = words_lower[index-1]

            if prev_word_lower.isdigit():
                estimated_counts[i] = int(prev_word_lower)
                continue

            if prev_word_lower in word_to_number:
                estimated_counts[i] = word_to_number[prev_word_lower]
                continue

            if prev_word_lower == "the":
                #assume implying the one item
                estimated_counts[i] = 1
                continue

            #if referring to item in singular and previous word doesn't specify count, assume implied 1
            estimated_counts[i] = 1



        elif (plural_items[i] in words_lower):
            index = words_lower.index(plural_items[i])
            if(index==0):
                #assume all items implied
                estimated_counts[i] = -1
                continue

            prev_word_lower = words_lower[index-1]

            if prev_word_lower.isdigit():
                estimated_counts[i] = int(prev_word_lower)
                continue

            if prev_word_lower in word_to_number:
                estimated_counts[i] = word_to_number[prev_word_lower]
                continue

            if prev_word_lower == "the" or prev_word_lower == "all":
                #assume implying all of the item
                estimated_counts[i] = -1
                continue

            #if referring to item in plural and previous word doesn't specify count, assume implied all
            estimated_counts[i] = -1

        else:
            #item was not mentioned
            continue

    #print("items = ", items)
    #print("final estimated counts = ", estimated_counts)
    return (estimated_counts)




def testFeatureExtraction():

    while(True):
        input_str = input('Enter a response: ')
        if (input_str == "quit"):
            break

        extractItemCounts(input_str)


def simpleTest(filepath):

    lines = utils.fileToList(filepath)

    assert((len(lines) % 2) == 0)

    for i in range(len(lines)//2):
        input_str = lines[2*i]
        gold = numpy.fromstring(lines[2*i + 1], dtype=int, sep=" ").tolist()

        estimated_counts = extractItemCounts(input_str)

        if(gold != estimated_counts):
            print("Error in input: ", input_str)
            print("Gold = ", countsToString(gold))
            print("Estimated = ", countsToString(estimated_counts))


#TODO: improve detecting the end of conversation
def terminatedConversation(line, words, words_lower):
    if ("done" in words_lower or "deal" in words_lower):
        return True
    else:
        return False

def terminatedConversation2(input_str):
    words_lower = tokenize(input_str.lower())

    if ("done" in words_lower or "deal" in words_lower):
        return True
    else:
        return False


#TODO: improve detection if opponent is offering a compromise
def checkCompromise(line, words, words_lower):
    givenItems, takenItems = None, None
    if ("if" in words_lower):
        index = words_lower.index("if")
        if (words_lower[index + 1] == "i"):
            #format of "you take these, if i take these"
            takenItems = extractItemCounts(" ".join(words[index+1:]))
            givenItems = extractItemCounts(" ".join(words[:index]))
        elif (words_lower[index + 1] == "you"):
            #format of "i'll take these if you take those"
            givenItems = extractItemCounts(" ".join(words[index+1:]))
            takenItems = extractItemCounts(" ".join(words[:index]))
        else:
            return None

        return givenItems, takenItems

    elif ("but" in words_lower):
        #not sure how to parse this
        return None
    else:
        return None


def testResponses(input_file):

    lines = utils.fileToList(input_file)

    for line in lines:
        print("Input = ", line)
        estimated_counts = extractItemCounts(line)
        words = tokenize(line)
        words_lower = [word.lower() for word in words]

        if(terminatedConversation(line, words, words_lower)):
            print("Are you suggesting we've come to a deal?")
            continue

        compromise = checkCompromise(line, words, words_lower)
        if(compromise != None):
            if compromise[0] == [0]*len(items) and compromise[1] == [0]*len(items):
                print("It seems like you wanted a compromise but I din't know what you want to give me " +
                    "or what you want in return")
                continue

            if compromise[0] == [0]*len(items):
                print("It seems like you wanted a compromise but didn't give me anything in return for " +
                    "you taking " + countsToString(compromise[1]))
                continue

            if compromise[1] == [0]*len(items):
                print("It seems like you wanted a compromise but didn't ask for anything in return for " +
                    "giving me " + countsToString(compromise[0]))
                continue

            print("Are you suggesting to give me " + countsToString(compromise[0]) + " in return for " +
                    "you taking " + countsToString(compromise[1]))
            continue

        '''if (input is a rejection of offer):
            print("You didn't want to take up my offer?")

        if (input is an affirmation of offer):
            print("You agree? I'll end the negotiation")

        if (input is a request):
            print("Are you asking for: " + countsToString(estimated_counts) + "?")'''

        print("Are you asking for: " + countsToString(estimated_counts) + "?")


def countsToString(counts):
    result = ""
    for i, item in enumerate(items):
        result += "{}=(count:{}), ".format(item, counts[i])
    #trim last comma and space
    return result[:-2]

def inputToString(input):
    result = ""
    for i, item in enumerate(items):
        result += "{}=(count:{} value:{}), ".format(item, input[i][0], input[i][1])
    #trim last comma and space
    return result[:-2]

def completeConversation(trainExamples):

    #import pdb; pdb.set_trace()

    total_negotations = 0

    total_bot_score = 0
    total_opponent_score = 0

    

    while(True):
        index = random.choice(range(len(trainExamples)))
        ex = trainExamples[index]

        bot_input = ex["input"]
        opponent_input = ex["partner input"]
        print("Opponent input = " + inputToString(opponent_input))

        total_counts = [count for count,value in bot_input]
        bot_values = [value for count,value in bot_input]
        opponent_values = [value for count,value in opponent_input]

        bot_value_sort = np.argsort((np.multiply(bot_values, total_counts)))

        mostValuableIndex = bot_value_sort[len(items) -1]
        middleIndex = bot_value_sort[1]
        leastValuableIndex = bot_value_sort[0]

        botItems, opponentItems = None, None

        attemptedTopItem = False
        num_attempts = 0


        botStarting = random.choice([True, False])
        if(botStarting):
            print("How about I take the " + plural_items[mostValuableIndex] + " and the rest " +
                        "is yours")
            botItems = [0]*len(items)
            botItems[mostValuableIndex] = total_counts[mostValuableIndex]
            opponentItems = total_counts[:]
            opponentItems[mostValuableIndex] = 0
        while(True):
            input_str = input('Enter a request: ')
            if(input_str == "<selection>"):
                break

            if(terminatedConversation2(input_str)):
                print("<selection>")
                break;

            if (num_attempts > 5):
                print("We are at an impasse, no deal")
                botItems = [0]*len(items)
                opponentItems = [0]*len(items)
                break;

            requested_counts = extractItemCounts(input_str)
            adjusted_requested_counts = requested_counts[:]
            remaining_counts = [0]*len(items)
            for i in range(len(items)):
                if requested_counts[i]==-1:
                    adjusted_requested_counts[i] = total_counts[i]
                    remaining_counts[i] = 0
                else:
                    remaining_counts[i] = max(total_counts[i] - requested_counts[i],0)

            if adjusted_requested_counts[mostValuableIndex] > 0:
                num_attempts += 1
                if not attemptedTopItem:
                    print("Not good, how about I take the " + plural_items[mostValuableIndex] + " and the rest " +
                        "is yours")
                    attemptedTopItem = True

                    botItems = [0]*len(items)
                    botItems[mostValuableIndex] = total_counts[mostValuableIndex]
                    opponentItems = total_counts[:]
                    opponentItems[mostValuableIndex] = 0
                else:
                    print("How about I take the " + plural_items[leastValuableIndex] + " and the " +
                         plural_items[middleIndex] + " and the rest is yours")
                    botItems = total_counts[:]
                    botItems[mostValuableIndex] = 0
                    opponentItems = [0]*len(items)
                    opponentItems[mostValuableIndex] = total_counts[mostValuableIndex]
            else:
                print("Okay, we have a deal")
                opponentItems = adjusted_requested_counts[:]
                botItems = remaining_counts[:]

        print("bot items = ", botItems)
        print("bot values = ", bot_values)
        total_negotations += 1
        total_bot_score += np.sum(np.multiply(botItems, bot_values))
        total_opponent_score += np.sum(np.multiply(opponentItems, opponent_values))


        print("Bot input:" + inputToString(bot_input))
        print("Final: bot items = " + countsToString(botItems))
        print("Final: opponent items = " + countsToString(opponentItems))
        print("Bot score = ", np.sum(np.multiply(botItems, bot_values)))
        print("Opponent score = ", np.sum(np.multiply(opponentItems, opponent_values)))


        #bot 55 -> 6.11 pts
        #me 85 -> 9.44 pts
        # total of 9 negotiations
        sentinel = input("Type 'quit' to exit, anything else to run another example: ")
        if (sentinel == 'quit'):
            break

    print("total negotiations = ", total_negotations)

    print("average score for bot = ", total_bot_score/float(total_negotations))
    print("average score for opponent = ", total_opponent_score/float(total_negotations))
    print("average score overall = ", (total_bot_score + total_opponent_score)/(total_negotations*2))


class chatBot:
    def __init__(self, bot_input):
        self.total_counts = [count for count,value in bot_input]
        self.bot_values = [value for count,value in bot_input]

        self.bot_value_sort = np.argsort((np.multiply(self.bot_values, self.total_counts)))

        self.mostValuableIndex = self.bot_value_sort[len(items) -1]
        self.middleIndex = self.bot_value_sort[1]
        self.leastValuableIndex = self.bot_value_sort[0]

        self.botItems, opponentItems = None, None

        self.attemptedTopItem = False
        self.num_attempts = 0

    def openNegotiation(self):
        responseBuffer = ("How about I take the " + plural_items[self.mostValuableIndex] + " and the rest " +
                        "is yours")
        self.botItems = [0]*len(items)
        self.botItems[self.mostValuableIndex] = self.total_counts[self.mostValuableIndex]
        self.opponentItems = self.total_counts[:]
        self.opponentItems[self.mostValuableIndex] = 0
        return responseBuffer

    def respond(self,input_str):

        if(terminatedConversation2(input_str)):
            return "<selection>"

        if (self.num_attempts > 5):
            print("We are at an impasse, no deal")
            self.botItems = [0]*len(items)
            self.opponentItems = [0]*len(items)
            return "<selection>"

        responseBuffer = None

        requested_counts = extractItemCounts(input_str)
        adjusted_requested_counts = requested_counts[:]
        remaining_counts = [0]*len(items)
        for i in range(len(items)):
            if requested_counts[i]==-1:
                adjusted_requested_counts[i] = self.total_counts[i]
                remaining_counts[i] = 0
            else:
                remaining_counts[i] = max(self.total_counts[i] - requested_counts[i],0)

        if adjusted_requested_counts[self.mostValuableIndex] > 0:
            self.num_attempts += 1
            if not self.attemptedTopItem:
                responseBuffer = ("Not good, how about I take the " + plural_items[self.mostValuableIndex] + " and the rest " +
                    "is yours")
                self.attemptedTopItem = True

                self.botItems = [0]*len(items)
                self.botItems[self.mostValuableIndex] = self.total_counts[self.mostValuableIndex]
                self.opponentItems = self.total_counts[:]
                self.opponentItems[self.mostValuableIndex] = 0
            else:
                responseBuffer = ("How about I take the " + plural_items[self.leastValuableIndex] + " and the " +
                     plural_items[self.middleIndex] + " and the rest is yours")
                self.botItems = self.total_counts[:]
                self.botItems[self.mostValuableIndex] = 0
                self.opponentItems = [0]*len(items)
                self.opponentItems[self.mostValuableIndex] = self.total_counts[self.mostValuableIndex]
        else:
            responseBuffer = ("Okay, we have a deal")
            self.opponentItems = adjusted_requested_counts[:]
            self.botItems = remaining_counts[:]

        return responseBuffer






def botToBot(trainExamples):

    while(True):
        index = random.choice(range(len(trainExamples)))
        ex = trainExamples[index]

        bot1_input = ex["input"]
        bot2_input = ex["partner input"]
        print("Bot1 input = " + inputToString(bot1_input))
        print("Bot2 input = " + inputToString(bot2_input))

        bot1 = chatBot(bot1_input)
        bot2 = chatBot(bot2_input)

        negotiatonOver = False
        bot1Starting = random.choice([True, False])
        bot1Response = None
        if(bot1Starting):
            bot1Response = bot1.openNegotiation()
        else:
            bot2Response = bot2.openNegotiation()
            print("Bot2: " + bot2Response)
            if bot2Response == "<selection>":
                negotiatonOver = True
            else:
                bot1Response = bot1.respond(bot2Response)

        if(not negotiatonOver):
            while(True):
                print("Bot1: " + bot1Response)
                if bot1Response == "<selection>":
                    break
                bot2Response = bot2.respond(bot1Response)
                print("Bot2: " + bot2Response)
                if bot2Response == "<selection>":
                    break
                bot1Response = bot1.respond(bot2Response)




        print("Final: bot1 items = " + countsToString(bot1.botItems))
        print("Final: bot2 items = " + countsToString(bot2.botItems))
        print("Bot1 score = ", np.sum(np.multiply(bot1.botItems, bot1.bot_values)))
        print("Bot2 score = ", np.sum(np.multiply(bot2.botItems, bot2.bot_values)))



        sentinel = input("Type 'quit' to exit, anything else to run another example: ")
        if (sentinel == 'quit'):
            break

def botToBotTest(trainExamples):

    print("Num examples = ", len(trainExamples))

    index = 0

    total_negotations = 0
    num_agree = 0

    total_starter_score = 0
    total_responder_score = 0


    while(index < len(trainExamples)):

        ex = trainExamples[index]

        bot1_input = ex["input"]
        bot2_input = ex["partner input"]
        #print("Bot1 input = " + inputToString(bot1_input))
        #print("Bot2 input = " + inputToString(bot2_input))

        bot1 = chatBot(bot1_input)
        bot2 = chatBot(bot2_input)

        negotiatonOver = False
        bot1Starting = random.choice([True, False])
        bot1Response = None
        if(bot1Starting):
            bot1Response = bot1.openNegotiation()
        else:
            bot2Response = bot2.openNegotiation()
            #print("Bot2: " + bot2Response)
            if bot2Response == "<selection>":
                negotiatonOver = True
            else:
                bot1Response = bot1.respond(bot2Response)

        if(not negotiatonOver):
            while(True):
                #print("Bot1: " + bot1Response)
                if bot1Response == "<selection>":
                    break
                bot2Response = bot2.respond(bot1Response)
                #print("Bot2: " + bot2Response)
                if bot2Response == "<selection>":
                    break
                bot1Response = bot1.respond(bot2Response)




        #print("Final: bot1 items = " + countsToString(bot1.botItems))
        #print("Final: bot2 items = " + countsToString(bot2.botItems))
        #print("Bot1 score = ", np.sum(np.multiply(bot1.botItems, bot1.bot_values)))
        #print("Bot2 score = ", np.sum(np.multiply(bot2.botItems, bot2.bot_values)))
        agreement = True
        for i in range(len(items)):
            if bot1.botItems[i] != bot2.opponentItems[i]:
                agreement = False
            if bot2.botItems[i] != bot1.opponentItems[i]:
                agreement = False

        if agreement:
            num_agree += 1

        bot1_score = np.sum(np.multiply(bot1.botItems, bot1.bot_values))
        bot2_score = np.sum(np.multiply(bot2.botItems, bot2.bot_values))

        if(bot1Starting):
            total_starter_score += bot1_score
            total_responder_score += bot2_score
        else:
            total_starter_score += bot2_score
            total_responder_score += bot1_score

        total_negotations += 1

        index += 1


    print("total negotiations = ", total_negotations)
    print("num agreed = ", num_agree)
    print("Agreement rate = ", num_agree/total_negotations)

    print("average score for starting bot = ", total_starter_score/float(total_negotations))
    print("average score for responding bot = ", total_responder_score/float(total_negotations))
    print("average score overall = ", (total_starter_score + total_responder_score)/(total_negotations*2))

def rnnBotConsole(trainExamples, vocab):

    #import pdb; pdb.set_trace()

    vocab.append("<START>")
    vocab.append("<sos>")
    vocab.append("<eos>")
    vocab.append("YOU")
    vocab.append("THEM")
    if "<UNK>" not in vocab:
        vocab.append("<UNK>")
    vocab = sorted(vocab)

    wordToIndex, indexToWord = utils.getIndexTranslations(vocab)


    #encoder = torch.load("savedModels/encoderTrained.pth")
    #decoder = torch.load("savedModels/decoderTrained.pth")
    #goals_encoder = torch.load("savedModels/goals_encoderTrained.pth")
    encoder = torch.load("savedModels/encoder.pth")
    decoder = torch.load("savedModels/decoder.pth")
    goals_encoder = torch.load("savedModels/goals_encoder.pth")
    output_classifier = torch.load("savedModels/outputClassifier.pth")



    while(True):
        index = random.choice(range(len(trainExamples)))
        ex = trainExamples[index]

        bot_input = ex["input"]
        opponent_input = ex["partner input"]
        print("\n\nOpponent input = " + inputToString(opponent_input))
        #print("Bot input = ", inputToString(bot_input))

        total_counts = [count for count,value in bot_input]
        bot_values = [value for count,value in bot_input]
        opponent_values = [value for count,value in opponent_input]

        '''bot_value_sort = np.argsort((np.multiply(bot_values, total_counts)))

        mostValuableIndex = bot_value_sort[len(items) -1]
        middleIndex = bot_value_sort[1]
        leastValuableIndex = bot_value_sort[0]

        botItems, opponentItems = None, None

        attemptedTopItem = False
        num_attempts = 0'''

        bot_goals_tensor = utils.toTensor(bot_input, "goals")
        encoded_goals = goals_encoder(bot_goals_tensor)

        running_context = ["<START>"]


        botStarting = random.choice([True, False])
        if(botStarting):

            #encode
            encoder_input = torch.tensor([wordToIndex["<START>"]], dtype=torch.long)[0]
            hidden = encoder.initHidden()
            hidden = encoder(encoder_input, hidden, encoded_goals)

            #decode
            selected_tokens = []

            encoded_context = hidden
            hidden = decoder.initHidden()
            decoder_input = torch.tensor([wordToIndex["<sos>"]], dtype=torch.long)[0]
            while(True):
                output, hidden = decoder(decoder_input, encoded_context, hidden, encoded_goals)

                #categorical = torch.distributions.Categorical(output)
                #selected_token = categorical.sample()
                selected_token_probability, selected_token = torch.max(output, 1)

                selected_token_as_word = indexToWord[int(selected_token[0].numpy())]

                if (selected_token_as_word == "<eos>"):
                    break

                #if(selected_token_as_word != "THEM" and selected_token_as_word != "YOU"):
                selected_tokens.append(selected_token_as_word)

                decoder_input = selected_token[0]

            print("Bot statement: ", " ".join(selected_tokens))
            running_context += ["YOU"] + selected_tokens + ["<eos>"]
            #print("running context = ", running_context)


        while(True):
            input_str = input('Enter a response: ')
            if(input_str == "<selection>"):
                running_context += ["THEM", "<selection>"]
                break

            tokenized = tokenize(input_str)

            running_context += ["THEM"] + tokenized + ["<eos>"]
            #print("running_context = ", running_context)

            running_context_idx = [wordToIndex[word] if word in wordToIndex else wordToIndex["<UNK>"] for word in running_context]
            running_context_tensor = torch.tensor(running_context_idx, dtype=torch.long)

            context = running_context_tensor
            hidden = encoder.initHidden()
            num_context_words = context.size(0)
            for i in range(num_context_words):
                hidden = encoder(context[i], hidden, encoded_goals)

            #decode
            maxResponseLen = 50
            selected_tokens = []
            encoded_context = hidden
            hidden = decoder.initHidden()
            decoder_input = torch.tensor([wordToIndex["<sos>"]], dtype=torch.long)[0]
            while(True):
                output, hidden = decoder(decoder_input, encoded_context, hidden, encoded_goals)

                selected_token_probability, selected_token = torch.max(output, 1)

                selected_token_as_word = indexToWord[int(selected_token[0].numpy())]

                if (selected_token_as_word == "<eos>"):
                    break

                #if(selected_token_as_word != "THEM" and selected_token_as_word != "YOU"):
                selected_tokens.append(selected_token_as_word)


                if(len(selected_tokens) >= maxResponseLen):
                    break

                decoder_input = selected_token[0]


            print("Bot response: ", " ".join(selected_tokens))
            running_context += ["YOU"] + selected_tokens + ["<eos>"]

            if (selected_tokens == ["<selection>"]):
                break

        #print("final running context = ", " ".join(running_context))
        running_context_idx = [wordToIndex[word] if word in wordToIndex else wordToIndex["<UNK>"] for word in running_context]
        running_context_tensor = torch.tensor(running_context_idx, dtype=torch.long)

        num_words = running_context_tensor.size(0)

        hidden = output_classifier.initHidden()
        for i in range(num_words):
            outputs, hidden = output_classifier(running_context_tensor[i], hidden, bot_goals_tensor)

        # in form: counts = [count1, count2, count3, opponent_count1, opponent_count2, opponent_count3]
        bot_selection = findBestValidPrediction(outputs, bot_input)
        bot_counts = bot_selection[:3]
        player_counts = bot_selection[3:]

        print("Bot input = ", inputToString(bot_input))
        print("Bot items: ", countsToString(bot_counts))
        print("Player items: ", countsToString(player_counts))

        print("Bot score = ", np.sum(np.multiply(bot_counts, bot_values)))
        print("Opponent score = ", np.sum(np.multiply(player_counts, opponent_values)))



        sentinel = input("Type 'quit' to exit, anything else to run another example: ")
        if (sentinel == 'quit'):
            break


if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    with open(args.train_vocab_json, "r") as fp:
        vocab = json.load(fp)

    #testFeatureExtraction()

    #simpleTest("testing/parseCountsTests.txt")

    #testResponses("testing/inputs.txt")

    #completeConversation(trainExamples)

    #botToBot(trainExamples)
    #botToBotTest(trainExamples)

    rnnBotConsole(trainExamples, vocab)


