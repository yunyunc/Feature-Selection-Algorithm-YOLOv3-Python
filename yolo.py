"""
Class definition of YOLO_v3 style detection model on image and video
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import colorsys
import os
from timeit import default_timer as timer
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from keras import models
from keras import layers

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        print(self.yolo_model.layers)
        #  my code here:
        print("Bingo! let's look into it!")
        print(self.yolo_model.layers[1])
        self.yolo_model.summary()
        print(self.yolo_model.get_layer("conv2d_1").output[0,0,0,1])
        for layer in self.yolo_model.layers:
            print(layer)
            print(layer.name)
        layer_dict = dict([(layer.name, layer) for layer in self.yolo_model.layers])

        filter_out = []
        filter_num_sum = 0
            # for i in range(75):
            # layer_name = "conv2d_%d" %(i+1)
        layer_name = "conv2d_70"
        print(layer_name)
        layer_output = layer_dict[layer_name].output
        print("this is the output")
        print(layer_output)
        print(layer_output[:,:,:,0])
        filter_shape = K.shape(layer_output)
        filter_num = filter_shape[3]
        #  loss = tf.math.divide (sum_val, tf.shape (abs_val))
        #  loss = K.mean(layer_output[:, :, :, filter_index])brooch        #  part1 of my code ends
        out_boxes, out_scores, out_classes, layer_output, filter_num = self.sess.run(
                                                           [self.boxes, self.scores, self.classes, layer_output, filter_num],
                                                           feed_dict={
                                                           self.yolo_model.input: image_data,
                                                           self.input_image_shape: [image.size[1], image.size[0]],
                                                           K.learning_phase(): 0
                                                           })
        print("filter_num",filter_num)
        print("this is the layer_output",layer_output)
        filter_num_sum += filter_num
        # filter_num is the index of the number of filter
        abs_val = K.abs(layer_output[:, :, :, 22])
        sum_val = K.sum(abs_val)
        sum_val = K.eval(sum_val)
        print("sum_val",sum_val)
        tensor_shape = K.shape(abs_val)
        sum_num = tensor_shape[1]*tensor_shape[2]
        sum_num = tf.to_float(sum_num, name='ToFloat')
        sum_num = K.eval(sum_num)
        print("sum_num", sum_num)
        loss = sum_val/sum_num
        print("loss", loss)
        filter_out.append(loss)
        print("filter_num_sum", filter_num_sum)
        print(filter_out)
#  x = np.arange(0, filter_num_sum, 1)
#  lines = plt.plot(x, filter_out, 'o')
#  plt.show()
        print(tensor_shape)
        print(tensor_shape[1])
        print(tensor_shape[2])
        print("filter_num: ", filter_num)
        # my code_2 here:
        # compute the gradient of the input picture wrt this loss
        # true_loss = K.mean(layer_output[:, :, :, filter_index])
        # grads = K.gradients(true_loss, self.yolo_model.input)[0]
        # layer_dict= dict([(layer.name, layer) for layer in self.yolo_model.layers])
        # layer_name = 'conv2d_1'
        # filter_index = 0
        # layer_output = layer_dict[layer_name].output
        # normalization trick: we normalize the gradient
        # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        # iterate = K.function([self.yolo_model.input], [true_loss, grads])

#   init = tf.global_variables_initializer()         # When init is run later(session.run(init)),
#   with tf.Session() as session:                    # Create a session and print the output
#            session.run(init)                            # Initializes the variables
#            print(session.run(loss))


        #input_data= mpimg.imread('/Users/chenyun/Downloads/7.png')
        #print("shape of input image: ", input_data.shape)
        #input_data = cv2.resize(input_data, (224, 224))
        #print(input_data.shape)
        #input_img_data[0,:,:,:]=input_data

# run gradient ascent for 500 steps
#        for i in range(500):
#            loss_value, grads_value = iterate([input_img_data])
#            input_img_data += grads_value * 10
#            print(loss_value)

# util function to convert a tensor into a valid image
#   def deprocess_image(x):
# normalize tensor: center on 0., ensure std is 0.1
#        x -= x.mean()
#        x /= (x.std() + 1e-5)
#        x *= 0.1
    
# clip to [0, 1]
#        x += 0.5
#        x = np.clip(x, 0, 1)
    
# convert to RGB array
#        x *= 255
#        x = x.transpose((1, 2, 0))
#        x = np.clip(x, 0, 255).astype('uint8')
#        return 0

#        img = input_data_converted[0]
#        img = deprocess_image(img)

# print(img.shape)
# img=np.transpose(img, (0, 2, 1))
# imgplot = plt.imshow(img)
#        print(img.shape)


# img = input_img_data[0]
# img_2 = deprocess_image(img)

# print(img_2.shape)
# img_2=np.transpose(img_2, (0, 2, 1))
# plt.savefig('img_2.png')
        #imgplot=plt.imshow(img_2)
# print(img_2.shape)


# part_2 code ends here


        layer_output = layer_dict[layer_name].output
        
        
        
        print('Found {} boxes for {}, hello'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300



        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            
    
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        
        return loss

    def close_session(self):
        self.sess.close()







def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

