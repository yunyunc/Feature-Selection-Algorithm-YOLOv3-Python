import sys
import argparse
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image
import os
import glob
from matplotlib import pyplot as plt

def detect_img(yolo):
    while True:
        image_list = []
        input_path = input('Input image filename:')
        input_path = input_path+'/*.jpg'
        print("input_path",input_path)
        loss_list = []
        n = 0
        for filename in glob.glob(input_path):
            print("filename",filename)
            img = Image.open(filename)
            loss = yolo.detect_image(img)
            print("I am telling you detecting one image!")
            #r_image.show()
            loss_list.append(loss)
            n=n+1
            print("this is the picture of:", n)
        print("loss_list:",loss_list)
        x = np.arange(0, 30, 1)
        lines = plt.plot(x, loss_list, 'o')
        plt.show()
        plt.hist(loss_list, bins='auto')
        plt.show()
        mean_list = sum(loss_list)/len(loss_list)
        loss_list= np.asarray(loss_list)
        variance = np.var(loss_list)
        print("mean is:", mean_list)
        print("variance is", variance)
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
