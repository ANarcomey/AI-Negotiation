import argparse



def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train-data', type=str,
						default = "data/train.txt",
						help = "File path for training data txt")

	parser.add_argument('--val-data', type=str,
						default = "data/val.txt",
						help = "File path for validation data txt")

	parser.add_argument('--test-data', type=str,
						default = "data/test.txt",
						help = "File path for testing data txt")

	parser.add_argument('--train-data-json', type=str,
						default = "data/train.json",
						help = "File path for training data json")

	parser.add_argument('--val-data-json', type=str,
						default = "data/val.json",
						help = "File path for validation data json")

	parser.add_argument('--test-data-json', type=str,
						default = "data/test.json",
						help = "File path for testing data json")



	args = parser.parse_args()

	return args