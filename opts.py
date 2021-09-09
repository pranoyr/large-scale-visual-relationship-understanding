import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weight_path', type=str,
						help='path to weight file')
	parser.add_argument('--image_path', type=str,
						help='input image path')
	parser.add_argument('--batch_size', type=int, default=4,
						help='batch size')
	parser.add_argument('--lr', type=float,
						help='learning rate')
	parser.add_argument('--weight_decay', type=float,
						help='weight rate')
	parser.add_argument('--num_workers', type=int, default=0,
						help='number of workers for data loaders')
	parser.add_argument('--max_iter', type=int, default=125446,
						help='number of iterations')
	parser.add_argument('--scheduler', type=str, default="multi_step",
						help='scheduler')
	parser.add_argument('--begin_iter', type=int, default=1,
						help='starting iteration')
	parser.add_argument('--images_dir', type=str,
						help='Directory of images')
	parser.add_argument('--video_path', type=str,
						help='sample video path')
	# parser.add_argument('--multi_gpu', action='store_true', help='Enables multiple GPU training')
	args = parser.parse_args()

	return args
