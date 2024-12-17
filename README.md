# AI-specialist
In []:
from google.colab import drive

In []:
drive.flush_and_unmount()  # 기존 마운트 해제

drive.mount('/content/drive')  # 다시 마운트

%cd /content/drive/MyDrive

%pwd

%cd /content/drive/MyDrive/yolov5

!git clone https://github.com/ultralytics/yolov5  # clone repo

%cd yolov5

%pip install -qr requirements.txt # install dependencies

!pip install Pillow==10.3

!mkdir -p Train/labels
!mkdir -p Train/images
!mkdir -p Val/labels
!mkdir -p Val/images

##데이터를 학습용 : 검증용 7:3 검증 데이터 만들기
import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_set(train_path, val_path, split_ratio=0.3):
  """
  Train 데이터의 일부를 Val로 이동
  """
  #필요한 디렉토리 생성
  os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
  os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

  #Train 이미지 리스트 가져오기
  train_images = os.listdir(os.path.join(train_path, 'images'))
  train_images = [f for f in train_images if f.endswith(('.jpg','.jpeg','.png'))]

  #Train 이미지를 Train, Val로 분할
  _, val_images = train_test_split(train_images, test_size=split_ratio, random_state=42)

  #Val로 파일 복사
  for img in val_images:
    #이미지 복사
    src_image = os.path.join(train_path, 'images', img)
    dst_image = os.path.join(val_path, 'images', img)
    shutil.copy(src_image, dst_image)
    # 라벨 파일 복사
    label_file = os.path.splitext(img)[0] + '.txt'
    src_label = os.path.join(train_path, 'labels', label_file)
    dst_label = os.path.join(val_path, 'labels', label_file)
    if os.path.exists(src_label):
      shutil.copy(src_label, dst_label)

  print(f"Validation set created with {len(val_images)} images.")

#실행
train_path = '/content/drive/MyDrive/yolov5/Train'
val_path = '/content/drive/MyDrive/yolov5/Val'

create_validation_set(train_path, val_path)

def check_dataset():
  train_path = '/content/drive/MyDrive/yolov5/Train'
  val_path = '/content/drive/MyDrive/yolov5/Val'

  #Train 데이터셋 확인
  train_images = len(os.listdir(os.path.join(train_path, 'images')))
  train_labels = len(os.listdir(os.path.join(train_path, 'labels')))

  #Val 데이터셋 확인
  val_images = len(os.listdir(os.path.join(val_path, 'images')))
  val_labels = len(os.listdir(os.path.join(val_path, 'labels')))

  print("Dataset status:")
  print(f"Train - Images: {train_images}, {train_labels}")
  print(f"Val - Images: {val_images}, Labels: {val_labels}")

check_dataset()

import torch
import os
from IPython.display import Image, clear_output

%pwd

import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

  def _preproc(image, output_height=512, output_width=512, resize_side=512):
      ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
      with eager_mode():
          h, w = image.shape[0], image.shape[1]
          scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
          resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
          cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
          return tf.squeeze(cropped_image)

  def Create_npy(imagespath, imgsize, ext) :
      images_list = [img_name for img_name in os.listdir(imagespath) if
                  os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
      calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            # 파일 크기가 정상적인지 확인
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB")  # RGBA 이미지 등 다른 형식이 있을 경우 강제로 RGB로 변환
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)

Create_npy('/content/drive/MyDrive/yolov5/Train/images', 512, 'jpg')

!python train.py  --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache

# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

!python detect.py --weights /content/drive/MyDrive/yolov5/yolov5/runs/train/exp5/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/yolov5/Train/images

#display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/drive/MyDrive/yolov5/yolov5/runs/detect/exp11/*.jpg')[:10]: #이미지 파일 형식에 맞춰 .png 또는 .jpg 등으로 수정

https://github.com/Justin020515/AI-specialist/issues/1#issue-2744737082
    display(Image(filename=imageName))
    print("\n")

!python detect.py --weights /content/drive/MyDrive/yolov5/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/video2.avi

https://github.com/Justin020515/AI-specialist/issues/1#issue-2744737082
