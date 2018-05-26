
import sys  
import os
import re

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
        count = input_arr[2*item_i]
        value = input_arr[2*item_i + 1]
        input_tuples.append((count, value))

    if(verbose):
        print("Input as tuples = {}".format(input_tuples))

    return input_tuples

def dialogue_to_string(dialogue):

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
    sentences = re.findall("(YOU|THEM): (.*?) <eos>", full_dialogue)

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
        if(count_str == "<disagree>"):
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
        if(count_str == "<disagree>"):
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

    outputs = [("YOU", agentOutputs), ("THEM", opponentOutputs)]

    return outputs

def partnerSelectionToString(selection_tuples):
    result = "Partner Selection: "
    for i, item in enumerate(items):
        result += "{}=(count:{} value:{}), ".format(item, selection_tuples[i][0], selection_tuples[i][1])
    #trim last comma and space
    return result[:-2]

def parsePartnerSelection(line, verbose = False):
    ### TODO: unify with parsing of input??? same formatting

    selection_matches = re.findall("<partner_input> (.*)? </partner_input>", line)
    assert(len(selection_matches) == 1)

    selection_str = selection_matches[0]
    selection_arr = numpy.fromstring(selection_str, dtype=int, sep=" ")
    assert(len(selection_arr) == INPUT_SIZE)
    if(verbose): 
        print("Partner selection as string = {}".format(selection_str))
        print("Partner selection as array = {}".format(selection_arr))

    ### formatted as: book=(count:x1 value:x2) hat=(count:x3 value:x4) ball=(count:x5 value:x6)
    selection_tuples = []

    for item_i in range(len(items)):
        count = selection_arr[2*item_i]
        value = selection_arr[2*item_i + 1]
        selection_tuples.append((count, value))

    if(verbose):
        print("Partner selection as tuples = {}".format(selection_tuples))

    return selection_tuples

def exploreTrainingData(filepath):

    num_examples_to_parse = 5
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

        input_tuples = parseInput(line, verbose = True)
        print(inputToString(input_tuples))
        example_dict["input"] = input_tuples

        dialogue = parse_dialogue(line, verbose = True)
        print(dialogue_to_string(dialogue))
        example_dict["dialogue"] = dialogue

        output = parseOutput(line, verbose = True)
        print("Output = {}".format(output))
        example_dict["output"] = output

        partnerSelection = parsePartnerSelection(line, verbose = True)
        print(partnerSelectionToString(partnerSelection))
        example_dict["partner selection"] = partnerSelection

        Examples.append(example_dict)



        

        



if __name__ == '__main__':

    args = parse_arguments()

    exploreTrainingData(args.train_data)

    

