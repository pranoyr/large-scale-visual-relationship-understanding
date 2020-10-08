import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str,
                        help='path to weight file')
    parser.add_argument('--image_path', type=str,
                        help='input image path')
    args = parser.parse_args()

    return args
