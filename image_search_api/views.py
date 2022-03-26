from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from .models import File
# import os
import numpy as np
from numpy.linalg import norm
import joblib as pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
#from tensorflow.keras.applications.MobileNet import MobileNetV2,preprocess_input
#from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA




class FileView(APIView):
    # query = File.objects.get()
    # serializer = FileSerializer
  
  def get(self, request, *args, **kwargs):
      files = File.objects.get()
      serializer = FileSerializer(files, many = True)
      return Response(serializer.data, status=status.HTTP_200_OK)
  parser_classes = (MultiPartParser, FormParser)

  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      print('The image is ',file_serializer.data['file']) 
      similar_images = findImage('./'+file_serializer.data['file'])
      data = 'http://localhost:8000/'+file_serializer.data['file']
      print('The similar image array is ', similar_images)
      return Response(similar_images, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)



def findImage(my_image):
  similar_images_array = []
  new_model = tf.keras.models.load_model('saved_model/my_model')

  loaded_model = pickle.load(open('saved_model/neighbors.pkl', 'rb'))

  batch_size = 64
  img_size = 224
  root_dir = 'medias/new_one'
  img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
  datagen = img_gen.flow_from_directory(root_dir,
                                        target_size=(img_size, img_size),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
  img_path = my_image
  input_shape = (img_size, img_size, 3)
  img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
  img_array = image.img_to_array(img)
  expanded_img_array = np.expand_dims(img_array, axis=0)
  preprocessed_img = preprocess_input(expanded_img_array)
  test_img_features = new_model.predict(preprocessed_img, batch_size=1)

  _, indices = loaded_model.kneighbors(test_img_features)
  filenames = [root_dir + '/' + s for s in datagen.filenames]
  def similar_images(indices):
    print('Number of indices',len(indices))
    # plt.figure(figsize=(15,10), facecolor='white')
   
    plotnumber = 1    
    for index in indices:
        if plotnumber<=len(indices) :
            ax = plt.subplot(2,4,plotnumber)
            similar_images_array.append('http://localhost:8000/'+filenames[index])
            print("The similar image is ",filenames[index])
            # plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')            
            plotnumber+=1
    # plt.tight_layout()
  print(indices.shape)
  similar_images(indices[0])
  return similar_images_array


 



