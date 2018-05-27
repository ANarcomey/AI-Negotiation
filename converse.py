import sys  
import os
import re
import json

import numpy

from config import parse_arguments
from readData import exampleToString, parseExample

items = ["book", "hat", "ball"]
plural_items = ["books", "hats", "balls"]


word_to_number = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
                    "both":2}


def extractFeatures(input_str):


    #surround punctuation marks with spaces
    input_str = re.sub("([?,.!;])", r' \1 ', input_str)
    input_str = re.sub('\s{2,}', ' ', input_str)

    words = input_str.split(" ")
    #print("words = ", words)

    estimated_counts = [0]*len(items)
    words_lower = [word.lower() for word in words]
    for i, item in enumerate(items):
        if (item in words_lower):
            index = words_lower.index(item)
            if(index==0):
                #assume one item implied
                estimated_counts[i]=1
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
                estimated_counts[i]=-1
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
            estimated_counts = -1

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

        extractFeatures(input_str)


def automaticTest(filepath):

    if not os.path.isfile(filepath):
       print("File path {} does not exist. Exiting...".format(filepath))
       sys.exit()

    lines = [line.rstrip('\n') for line in open(filepath)]

    assert((len(lines) % 2) == 0)

    for i in range(len(lines)//2):
        input_str = lines[2*i]
        gold = numpy.fromstring(lines[2*i + 1], dtype=int, sep=" ").tolist()

        estimated_counts = extractFeatures(input_str)

        if(gold != estimated_counts):
            print("Error in input: ", input_str)
            print("Gold = ", gold)
            print("Estimated = ", estimated_counts)









if __name__ == '__main__':

    args = parse_arguments()

    with open(args.train_data_json, "r") as fp:
        trainExamples = json.load(fp)

    #testFeatureExtraction()

    automaticTest("testing/parseCountsTests.txt")