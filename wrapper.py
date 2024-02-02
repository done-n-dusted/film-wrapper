import argparse
import tensorflow as tf
import tensorflow_hub as hub

import requests
import numpy as np

from typing import Generator, Iterable, List, Tuple
import mediapy as media

class Wrapper:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/film/1")
    
    def input_image(self, img1: str, img2: str):
        self.img1 = media.read_image(img1)
        self.img2 = media.read_image(img2)

    def predict_mid(self, out, save=True):
        input = {
            'time': np.expand_dims(0.5, axis=0),
            'x0': np.expand_dims(self.img1, axis=0),
            'x1': np.expand_dims(self.img2, axis=0)
        }

        mid_frame = self.model(input)
        img_mid = mid_frame['image'][0].numpy()
        if save:
            media.write_image('output/images/' + out + '.png', img_mid)
            media.write_video('output/videos/' + out + '.mp4', [self.img1, img_mid, self.img2], fps=30)
        
        return img_mid

    def predict_interpolate(self, out, n){
        time
    
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description='Wrapper for the main function')
    parser.add_argument('--img1', type=str, help='Input image 1 location')
    parser.add_argument('--img2', type=str, help='Input image 2 location')
    parser.add_argument('--out', type=str, help='Name of the output file. Do not include extension.')
    parser.add_argument('--n', type=int, help='Number of frames to interpolate between img1 and img2', default=1)


    return parser.parse_args()

def main():
    args = parse_arguments()
    wrapper = Wrapper()
    wrapper.input_image(args.img1, args.img2)
    if args.n == 1:
        mid_frame = wrapper.predict_mid(args.out, True)
        print("Mid frame generated and saved")
    else:
        # wrapper.predict_interpolate(args.out, args.n)
        print("Interpolation generated and saved")

if __name__ == "__main__":
    main()

