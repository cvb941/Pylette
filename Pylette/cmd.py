import argparse

from Pylette import extract_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='path to image file', type=str)
    parser.add_argument('--n', help='the number of colors to extract',
                        type=int, default=5)
    parser.add_argument('--sort_by', help='sort by luminance or frequency', default='luminance', type=str)
    parser.add_argument('--stdout', help='whether to display the extracted color values in the stdout', type=bool,
                        default=True)
    parser.add_argument('--colorspace', help='which color-space to represent the colors in', default='RGB', type=str)
    parser.add_argument('--out_filename', help='where to save the csv file', default=None, type=str)
    args = parser.parse_args()
    palette = extract_colors(args.filename, palette_size=args.n, sort_mode=args.sort_by)

    palette.to_csv(filename=args.out_filename, frequency='True')


if __name__ == '__main__':
    main()
