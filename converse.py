import sys  
import os
import re
import json

import numpy

from config import parse_arguments
from readData import exampleToString, parseExample
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


#TODO: improve detection if opponent is offering a compromise
def checkCompromise(line, words, words_lower):
    givenItems, takenItems = None, None
    if ("if" in words_lower):
        index = words_lower.index("if")
        if (words_lower[index + 1] == "i"):
            #format of "you take these, if i take these"
            takenItems = extractItemCounts(words[index+1:].join(" "))
            givenItems = extractItemCounts(words[:index].join(" "))
        elif (words_lower[index + 1] == "you"):
            #format of "i'll take these if you take those"
            givenItems = extractItemCounts(words[index+1:].join(" "))
            takenItems = extractItemCounts(words[:index].join(" "))
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
                print("It seems like you wanted a compromise but I din't know what you want to give me " \
                    "or what you want in return")
                continue

            if compromise[0] == [0]*len(items):
                print("It seems like you wanted a compromise but didn't give me anything in return for " \
                    "you taking " + countsToString(compromise[1]))
                continue

            if compromise[1] == [0]*len(items):
                print("It seems like you wanted a compromise but didn't ask for anything in return for " \
                    "giving me " + countsToString(compromise[0]))
                continue

            print("Are you suggesting to give me " + countsToString(compromise[0]) + " in return for " /
                    "you taking " + countsToString(compromise[1]))
            continue

        '''if (input is a rejection of offer):
            print("You didn't want to take up my offer?")

        if (input is an affirmation of offer):
            print("You agree? I'll end the negotiation")

        if (input is a request):
            print("Are you asking for: " + countsToString(estimated_counts) + "?")'''










if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    #testFeatureExtraction()

    simpleTest("testing/parseCountsTests.txt")

    testResponses("testing/inputs.txt")


