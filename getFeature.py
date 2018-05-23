#!/usr/bin/python
import sys, os
import numpy as np
from sklearn.externals import joblib
import time

# local
import ext_utils

# Make sure that caffe is on the python path:
caffe_root = '/home/s02/fyk/frcnn'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

class CaffeExtractor:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, mean_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        # read binaryproto file to blob
        mean_blob = caffe.proto.caffe_pb2.BlobProto()
        with open(mean_file, 'rb') as f:
            mean_blob.ParseFromString(f.read())

        # convert blob to numpy.array
        mean_npy = np.array(caffe.io.blobproto_to_array(mean_blob))
        mean_pixel_npy = mean_npy[0].mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        # print(mean_pixel_npy) # this will get [102.71699095 115.77261497 123.5093643], a little diff from [104, 117, 123] for this is 224-cropped

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        #self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        self.transformer.set_mean('data', mean_pixel_npy) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # pyramid bins region num:
        self.bins = np.array([[1,1], [2,2], [3,3]])
        self.numRegions = np.sum(np.prod(self.bins, axis=1))
        #assert(14 == self.numRegions)
        self.numFeatures = 4096
    def extractFeature(self, imageListFile, image_base_path, batch_size=1):
        '''batch_size is of image file num'''
        r_batch = batch_size * self.numRegions
        self.net.blobs['data'].reshape(r_batch, 3, self.image_resize, self.image_resize)
        featureVectors = []
        cnt = 0
        batch_idx = 0
        tic = time.time()
        image_file_list = self.readImageList(imageListFile, image_base_path)
        N = len(image_file_list)
        for image_file in image_file_list:
            cnt += 1
            image = caffe.io.load_image(image_file)
            regions = ext_utils.makePyramidRegions(image, self.bins)
            #print(len(regions))
            for i in range(self.numRegions):
                transformed_image = self.transformer.preprocess('data', regions[i])
                self.net.blobs['data'].data[batch_idx] = transformed_image
                #self.net.blobs['data'].data[...] = transformed_image
                #_ = self.net.forward()
                if batch_idx == r_batch - 1:
                    self.net.forward()
                    feature = np.array(self.net.blobs['fc7'].data).reshape((batch_size,-1))
                    #print feature
                    # L2-normalized feature vectors
                    feature = feature / np.sqrt(np.sum(feature**2, axis=1))[:,np.newaxis]
                    featureVectors.extend(feature)
                    if N - cnt < batch_size:
                        batch_size = N - cnt
                        r_batch = batch_size * self.numRegions
                        self.net.blobs['data'].reshape(r_batch, 3, self.image_resize, self.image_resize)

                    batch_idx = 0
                else:
                    batch_idx += 1

            if cnt % 1000 == 0:
                toc = time.time()
                print("---------- %d (%d ms) -----------"%(cnt, (toc-tic) * 1000))
                tic = toc
            #if cnt == 160: break
        return featureVectors
        
    def makeLinearKernel(self, featureVectors):
        # Computing linear kernel for all samples
        K = featureVectors.dot(featureVectors.T)
        #joblib.dump(K, 'K-tmp.pkl')
        joblib.dump(K, 'feature/K.pkl', compress=3)
        return K

    def readImageList(self, imageListFile, image_base_path):
        imageList = []
        with open(imageListFile,'r') as fi:
            while(True):
                line = fi.readline().strip().split()# every line is a image file name
                if not line:
                    break
                imageList.append(image_base_path + line[0] + '.jpg')
        print('total %d images to process.'%(len(imageList)))
        return imageList

if __name__ == "__main__":
    gpu_id = 0
    model_def = 'VGG-f/VGG_CNN_F_deploy.prototxt'
    model_weights = 'VGG-f/VGG_CNN_F.caffemodel'
    mean_file = 'VGG-f/VGG_mean.binaryproto'
    image_resize = 224
    image_base_path = '/home/s02/fyk/VOCdevkit/VOC2012/JPEGImages/'
    imageListFile = 'trainval.txt'
    extractor = CaffeExtractor(gpu_id, model_def, model_weights, image_resize, mean_file)
    featureVectors = extractor.extractFeature(imageListFile, image_base_path, batch_size=128)
    fv = np.array(featureVectors)
    print(fv.shape)
    joblib.dump(featureVectors, 'feature/features-pyramid-vgg-f.pkl', compress=0) # there has bug when dump big object when compressing with level 3
    extractor.makeLinearKernel(fv)

