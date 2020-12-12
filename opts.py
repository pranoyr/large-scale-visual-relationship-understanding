import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weight_path', type=str,
						help='path to weight file')
	parser.add_argument('--image_path', type=str,
						help='input image path')
	parser.add_argument('--batch_size', type=int, default=2,
						help='batch size')
	parser.add_argument('--num_workers', type=int, default=4,
						help='number of workers for data loaders')
	parser.add_argument('--n_epochs', type=int, default=100,
						help='number of epochs')
	parser.add_argument('--scheduler', type=str, default="plateau",
						help='number of epochs')
	args = parser.parse_args()

	return args
