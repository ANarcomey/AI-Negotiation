import argparse



def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train-data', type=str,
						default = "data/train.txt",
						help = "File path for training data")

	parser.add_argument('--val-data', type=str,
						default = "data/val.txt",
						help = "File path for validation data")

	parser.add_argument('--test-data', type=str,
						default = "data/test.txt",
						help = "File path for testing data")



	args = parser.parse_args()

	return args