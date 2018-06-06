import argparse



def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train-data', type=str,
						default = "data/train.txt",
						help = "File path for training data as txt")

	parser.add_argument('--val-data', type=str,
						default = "data/val.txt",
						help = "File path for validation data as txt")

	parser.add_argument('--test-data', type=str,
						default = "data/test.txt",
						help = "File path for testing data as txt")

	parser.add_argument('--train-data-json', type=str,
						default = "data/train.json",
						help = "File path for training data as json")

	parser.add_argument('--val-data-json', type=str,
						default = "data/val.json",
						help = "File path for validation data as json")

	parser.add_argument('--test-data-json', type=str,
						default = "data/test.json",
						help = "File path for testing data as json")
	
	parser.add_argument('--train-vocab-json', type=str,
						default = "data/train-vocab.json",
						help = "File path for training set vocabulary as json")



	args = parser.parse_args()

	return args