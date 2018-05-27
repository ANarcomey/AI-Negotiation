
import sys  
import os
import re
import json

import numpy

from config import parse_arguments


items = ["book", "hat", "ball"]
INPUT_SIZE = 6

# defining each pair of numbers in input as (count, value)
COUNT_INDEX = 0
VALUE_INDEX = 1

def inputToString(input_tuples):
    result = "Input: "
    for i, item in enumerate(items):
        result += "{}=(count:{} value:{}), ".format(item, input_tuples[i][0], input_tuples[i][1])
    #trim last comma and space
    return result[:-2]

def parseInput(line, verbose = False):

    input_matches = re.findall("<input>(.*)?</input>", line)
    assert(len(input_matches) == 1)

    input_str = input_matches[0]
    input_arr = numpy.fromstring(input_str, dtype=int, sep=" ")
    assert(len(input_arr) == INPUT_SIZE)
    if(verbose): 
        print("Input as array = {}".format(input_arr))

    ### formatted as: book=(count:x1 value:x2) hat=(count:x3 value:x4) ball=(count:x5 value:x6)
    input_tuples = []

    for item_i in range(len(items)):
        count = int(input_arr[2*item_i])
        value = int(input_arr[2*item_i + 1])
        input_tuples.append((count, value))

    if(verbose):
        print("Input as tuples = {}".format(input_tuples))

    return input_tuples

def dialogueToString(dialogue):

    result = "\nDialogue sequence:\n"
    for sentence in dialogue:
        speaker = sentence[0]
        utterance = sentence[1]
        result += "Speaker = {}, Utterance = \"{}\"\n".format(speaker, utterance)

    return result


def parse_dialogue(line, verbose = False):
    dialogue_matches = re.findall("<dialogue>(.*)?</dialogue>", line)
    assert(len(dialogue_matches) == 1)

    full_dialogue = dialogue_matches[0]
    #each utterance ends in <eos>, last utterance is <selection> which ends conversation and
    #marks the end of dialogue string
    sentences = re.findall("(YOU|THEM): (.*?) (?:<eos>|$)", full_dialogue)

    if (verbose):
        print("\nDialogue match = {}".format(dialogue_matches))
        print("\nRegex found sentences = \n{}".format(sentences))

    dialogue = []

    for s in sentences:

        #for speaker and utterance pair
        assert(len(s) == 2)

        speaker = s[0]
        utterance = s[1]
        dialogue.append((speaker, utterance))

    return dialogue

def outputToString(output):
    result = "Output: "
    result += "\nYOU: "
    agentOutput = output[0]
    for i, item in enumerate(items):
        result += "{}=(count:{}), ".format(item, agentOutput[i])
    #trim last comma and space
    result = result[:-2]

    result += "\nTHEM: "
    opponentOutput = output[1]
    for i, item in enumerate(items):
        result += "{}=(count:{}), ".format(item, opponentOutput[i])
    #trim last comma and space
    result = result[:-2]
    return result + "\n"

def parseOutput(line, verbose = False):
    output_matches = re.findall("<output> (.*)? </output>", line)
    assert(len(output_matches) == 1)

    output_str = output_matches[0]
    output_arr = output_str.split(" ")

    if(verbose):
        print("Parsing output:")

    agentOutputs = []
    player = "YOU"
    for i in range(len(items)):

        count_str = output_arr[i]
        if(count_str == "<disagree>" or count_str == "<no_agreement>"):
            count = -1
        else:
            count_matches = re.findall("item\d+=(\d+)", count_str)
            assert (len(count_matches) == 1)
            count = count_matches[0]

        agentOutputs.append(count)


        if(verbose):
            print("Agent = {}, count as string = \"{}\", count = {}".format(player, count_str, count))


    opponentOutputs = []
    player = "THEM"
    for i in range(len(items)):
        count_str = output_arr[len(items) + i]
        if(count_str == "<disagree>" or count_str == "<no_agreement>"):
            count = -1
        else:
            count_matches = re.findall("item\d+=(\d+)", count_str)
            assert (len(count_matches) == 1)
            count = count_matches[0]

        opponentOutputs.append(count)

        if(verbose):
            print("Agent = {}, count as string = \"{}\", count = {}".format(player, count_str, count))


    if(verbose):
        print("Output as string = \"{}\"".format(output_str))
        print("Output as array = {}".format(output_arr))

    outputs = [agentOutputs, opponentOutputs]

    return outputs

def partnerInputToString(input_tuples):
    result = "Partner Input: "
    for i, item in enumerate(items):
        result += "{}=(count:{} value:{}), ".format(item, input_tuples[i][0], input_tuples[i][1])
    #trim last comma and space
    return result[:-2]

def parsePartnerInput(line, verbose = False):
    ### TODO: unify with parsing of input??? same formatting

    input_matches = re.findall("<partner_input> (.*)? </partner_input>", line)
    assert(len(input_matches) == 1)

    input_str = input_matches[0]
    input_arr = numpy.fromstring(input_str, dtype=int, sep=" ")
    assert(len(input_arr) == INPUT_SIZE)
    if(verbose): 
        print("Partner Input as array = {}".format(input_arr))

    ### formatted as: book=(count:x1 value:x2) hat=(count:x3 value:x4) ball=(count:x5 value:x6)
    input_tuples = []

    for item_i in range(len(items)):
        count = int(input_arr[2*item_i])
        value = int(input_arr[2*item_i + 1])
        input_tuples.append((count, value))

    if(verbose):
        print("Partner Input as tuples = {}".format(input_tuples))

    return input_tuples
    

def exploreTrainingData(filepath):

    num_examples_to_parse = 20
    Examples = []

    if not os.path.isfile(filepath):
       print("File path {} does not exist. Exiting...".format(filepath))
       sys.exit()

    fp = open(filepath, 'r') 

    for i,line in enumerate(fp):
        if (i >= num_examples_to_parse):
            break

        example_dict = {}

        print("\n\n\nline {} contents:\n {}".format(i, line))

        example_dict = parseExample(line)

        Examples.append(example_dict)

    with open("data/train.json", "w") as fp_json:
        json.dump(Examples, fp_json)

def parseExample(line):
    example_dict = {}
    input_tuples = parseInput(line, verbose = False)
    #print(inputToString(input_tuples))
    example_dict["input"] = input_tuples

    dialogue = parse_dialogue(line, verbose = False)
    #print(dialogueToString(dialogue))
    example_dict["dialogue"] = dialogue

    output = parseOutput(line, verbose = True)
    #print(outputToString(output))
    example_dict["output"] = output

    partnerInput = parsePartnerInput(line, verbose = False)
    #print(partnerInputToString(partnerInput))
    example_dict["partner input"] = partnerInput

    return example_dict

        
def exampleToString(example):
    result = "Example as string:\n"
    result += inputToString(example["input"])
    result += dialogueToString(example["dialogue"])
    result += outputToString(example["output"])
    result += partnerInputToString(example["partner input"])
    return result



if __name__ == '__main__':

    args = parse_arguments()

    exploreTrainingData(args.train_data)

    ##to load json file:
    with open(args.train_data_json, "r") as fp:
        Examples = json.load(fp)

    print("\n\n\n\n\n Displaying training examples loaded from json:")
    for ex in Examples:
        print(ex)
        print("\n")
        print(exampleToString(ex))
        print("\n\n\n\n")


    

