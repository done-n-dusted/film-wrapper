import argparse
import tensorflow as tf
import tensorflow_hub as hub
import time
import requests
import numpy as np

from typing import Generator, Iterable, List, Tuple
import mediapy as media
from PIL import Image

class Wrapper:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/film/1")
    
    def input_image(self, img1: str, img2: str):
        # self.img1 = media.read_image(img1) 
        # self.img2 = media.read_image(img2)
        self.img1 = np.array(Image.open(img1))/255.0
        self.img1 = tf.cast(self.img1, dtype=tf.float32).numpy()
        self.img2 = np.array(Image.open(img2))/255.0
        self.img2 = tf.cast(self.img2, dtype=tf.float32).numpy()

    def predict_mid(self, out, save=True):
        input = {
            'time': np.expand_dims(np.array([0.5], dtype = np.float32), axis=0),
            'x0': np.expand_dims(self.img1, axis=0),
            'x1': np.expand_dims(self.img2, axis=0)
        }

        mid_frame = self.model(input)
        img_mid = mid_frame['image'][0].numpy()
        if save:
            media.write_image('output/images/' + out + '.png', img_mid)
            media.write_video('output/videos/' + out + '.mp4', [self.img1, img_mid, self.img2], fps=30)
        
        return img_mid

    def _pred_mid(self, img1, img2):
        input = {
            'time': np.expand_dims(0.5, axis=0),
            'x0': np.expand_dims(img1, axis=0),
            'x1': np.expand_dims(img2, axis=0)
        }

        mid_frame = self.model(input)
        img_mid = mid_frame['image'][0].numpy()
        return img_mid
    
    def predict_interpolate(self, out, n, save=True):
        # call pred_mid appropiate so that total number of frames is n
        # save the frames
        frames = {x: None for x in range(n)}
        frames[0] = self.img1
        frames[n-1] = self.img2
        
        def dfs(l, r):
            if l + 1 == r:
                return
            mid = (l + r) // 2
            frames[mid] = self._pred_mid(frames[l], frames[r])
            dfs(l, mid)
            dfs(mid, r)
        
        dfs(0, n-1)
        frame_list = [frames[i] for i in range(n)]

        if save:
            media.write_video('output/videos/' + out + '.mp4', frame_list, fps=30)
        
        return frame_list
    



def parse_arguments():
    parser = argparse.ArgumentParser(description='Wrapper for the main function')
    parser.add_argument('--img1', type=str, help='Input image 1 location')
    parser.add_argument('--img2', type=str, help='Input image 2 location')
    parser.add_argument('--out', type=str, help='Name of the output file. Do not include extension.', default="result")
    parser.add_argument('--n', type=int, help='Number of frames to interpolate between img1 and img2', default=1)
    parser.add_argument('--save', type=bool, help='Save the output video or not', default=True)


    return parser.parse_args()

def main():
    start = time.time()
    args = parse_arguments()
    wrapper = Wrapper()
    wrapper.input_image(args.img1, args.img2)
    print("Images loaded. Time taken: ", time.time() - start, " seconds.")
    saved = ""
    if args.save:
        saved = " and saved"
    if args.n == 1:
        mid_frame = wrapper.predict_mid(args.out, args.save)
        print("Mid frame generated " + saved + ". Time taken: ", time.time() - start, " seconds.")
    else:
        wrapper.predict_interpolate(args.out, args.n, args.save)
        print("Interpolation generated " + saved + ". Time taken: ", time.time() - start, " seconds.")

if __name__ == "__main__":
    main()

