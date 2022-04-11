from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from .models import File
import os
import numpy as np
from numpy.linalg import norm
import joblib as pickle
from tqdm import tqdm
import math
from PIL import Image
import PIL
from pathlib import Path
import tensorflow as tf
import pickle 
import shutil

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
from numpy.lib.function_base import append
from re import I


class FileView(APIView):
    # query = File.objects.get()
    # serializer = FileSerializer
  parser_classes = (MultiPartParser, FormParser)
  def get(self, request, *args, **kwargs):
      files = File.objects.get()
      serializer = FileSerializer(files, many = True)
      return Response(serializer.data, status=status.HTTP_200_OK)
 

  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      print('The image is ',file_serializer.data['file']) 
      similar_images = findImage('./'+file_serializer.data['file'])
      data = 'http://localhost:8000'+file_serializer.data['file']
      return Response(similar_images, status=status.HTTP_201_CREATED)
    else:
      print('file serializer error', file_serializer.errors)
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)



def findImage(my_image):
  similar_images_array = []
  new_model = tf.keras.models.load_model('saved_model/my_model')

  loaded_model = pickle.load(open('saved_model/neighbors.pkl', 'rb'))

  batch_size = 64
  img_size = 224
  root_dir = 'media/new_one'
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



class TrainImage(APIView):
  parser_classes = (MultiPartParser, FormParser)

  def get(self, request, *args, **kwargs):
    return Response(data="image",status=status.HTTP_200_OK)
  

  def post(self, request, *args, **kwargs ):
    file_serializer = FileSerializer(data=request.data)
    delete_model()
    if file_serializer.is_valid():
      file_serializer.save()

      print("Imag is",file_serializer.data['file'])
      image = file_serializer.data['file'][1:]
      print(image)

      move_image(image)

     

      
      # similar_images = findImage('./'+file_serializer.data['file'])
      # data = 'http://localhost:8000'+file_serializer.data['file']
      img_size =224
      model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')
      batch_size = 64
      root_dir = 'media/new_one'

      img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
      datagen = img_gen.flow_from_directory(root_dir,
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              class_mode=None,
                                              shuffle=False)
      num_images = len(datagen.filenames)
      num_epochs = int(math.ceil(num_images / batch_size))
      feature_list = model.predict_generator(datagen, num_epochs)
      create_dir()
      model.save('saved_model/my_model') 
      neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='ball_tree',
                             metric='euclidean')
      neighbors.fit(feature_list)
      save_model(neighbors)
      # model.save()
      print("Num images   = ", len(datagen.classes))
      print("Shape of feature_list = ", feature_list.shape)
     
      return Response(file_serializer.data['file'], status=status.HTTP_201_CREATED)

    else:
      print('file serializer error', file_serializer.errors)
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
   
    # return Response(data=True,status=status.HTTP_201_CREATED)

 



def move_image(image):
  if os.path.exists(image):
    shutil.move(image, 'media/new_one/Faces')
    print("The file is moved")
  else:
    print('File does not exist')



def delete_model():
  shutil.rmtree('saved_model')


def save_model(neighbors):
  #Its important to use binary mode 
  knnPickle = open('saved_model/neighbors.pkl', 'wb') 

  # # # source, destination 
  pickle.dump(neighbors, knnPickle)   
  knnPickle.close();      
  print('neighbor dumped')



#creates a directory without throwing an error
def create_dir():
  if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
    print("Created Directory : ", 'saved_model')
  else:
    print("Directory already existed : ", 'saved_model')
  return dir

