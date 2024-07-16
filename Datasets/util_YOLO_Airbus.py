import os
import ast
import numpy as np
from PIL import Image
import math
import pandas as pd
import shutil
from random import shuffle
from tqdm import tqdm


def split_database(original_path, final_path, training_rate=0.5, valid_rate=0.3, test_rate=0.2):
    """
    Splits the database into training, validation, and test sets.

    :param original_path: Path to the original dataset
    :param final_path: Path to save the split datasets
    :param training_rate: Proportion of data to be used for training
    :param valid_rate: Proportion of data to be used for validation
    :param test_rate: Proportion of data to be used for testing
    """
    if training_rate + valid_rate + test_rate > 1.0:
        raise ValueError("The sum of training_rate, valid_rate, and test_rate must be 1")

    # Get all image files in the original path
    files = os.listdir(original_path)
    files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Shuffle the files
    shuffle(files)

    # Split the files into training, validation, and test sets
    n_train = int(len(files) * training_rate)
    n_valid = int(len(files) * valid_rate)
    n_test = len(files) - n_train - n_valid

    train_files = files[:n_train]
    valid_files = files[n_train:n_train + n_valid]
    test_files = files[n_train + n_valid:]

    # Create destination folders if they don't exist
    os.makedirs(os.path.join(final_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(final_path, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(final_path, 'test'), exist_ok=True)

    # Move files to the respective folders
    def move_files(file_list, category):
        category_path = os.path.join(final_path, category)
        for file in file_list:
            shutil.copy(os.path.join(original_path, file), category_path)
        print(f"Moved {len(file_list)} files to {category_path}")

    print('## TRAIN ##')
    move_files(train_files, 'train')

    print('## VALID ##')
    move_files(valid_files, 'valid')

    print('## TEST ##')
    move_files(test_files, 'test')

def extract_image_name(folder):
    """
    Extracts the names of all files in the specified folder.

    :param folder: Path to the folder
    :return: List of file names in the folder
    """
    return [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]


def build_YOLO_folders(path, name_fodler):
   path_YOLO = os.path.join(path, name_fodler)
   os.makedirs(os.path.join(path_YOLO, 'images/train'), exist_ok=True)
   os.makedirs(os.path.join(path_YOLO, 'labels/train'), exist_ok=True)
   os.makedirs(os.path.join(path_YOLO, 'images/test'), exist_ok=True)
   os.makedirs(os.path.join(path_YOLO, 'labels/test'), exist_ok=True)
   os.makedirs(os.path.join(path_YOLO, 'images/valid'), exist_ok=True)
   os.makedirs(os.path.join(path_YOLO, 'labels/valid'), exist_ok=True)
   
   print(f'YOLO Folders created in {path_YOLO}')

def crop_image(folder, img_name, x1, y1, x2, y2):
  # Crop the image: is going to return the image cropped, i.e., the image with the dimensions x1, y1, x2, y2
  # img_name: absolute path of the image
  # x1, y1 positions of the top, left of the image
  # crop_size: size of the crop image
  # img_crop: the return image
  img = Image.open(os.path.join(folder, img_name))
  img_crop = img.crop((x1,y1, x2, y2))
  return img_crop

def filter_points(df_points, x1, y1, x2, y2):
  # filter_points: return the points of the df that are inside x1, y1, x2, y2 or in the edge, i.e., at least one of the points is inside the block
  # df of points, where we have as columns idx, x1_bb, y1_bb, x2_bb, y2_bb, where _bb refers to the boundary boxes of the corresponding objects
  # x1, y1 and x2, y2, are the limits of the block of the subdividing image
  # points_inside_block return the points that are completely inside the image
  # points_in_edge_block return are the points of the objects that are cut as a result of the dividing image
  points_inside_block = df_points[(df_points['x1_bb'] >= x1) & (df_points['y1_bb'] >= y1) & (df_points['x2_bb'] <= x2) & (df_points['y2_bb'] <= y2)]
  points_in_edge_block = df_points[((((df_points['x1_bb'] >= x1) & (df_points['x1_bb'] <= x2)) & ((df_points['y1_bb'] >= y1) & ((df_points['y1_bb'] <= y2)) | ((df_points['y2_bb'] >= y1) & (df_points['y2_bb'] <= y2))))  |
                      ((df_points['x2_bb'] >= x1) & ((df_points['x2_bb'] <= x2)) & ((df_points['y1_bb'] >= y1) & ((df_points['y1_bb'] <= y2)) | ((df_points['y2_bb'] >= y1) & (df_points['y2_bb'] <= y2)))) |
                      ((df_points['y1_bb'] >= y1) & ((df_points['y1_bb'] <= y2)) & ((df_points['x1_bb'] >= x1) & ((df_points['x1_bb'] <= x2)) | ((df_points['x2_bb'] >= x1) & (df_points['x2_bb'] <= x2)))) |
                      ((df_points['y2_bb'] >= y1) & ((df_points['y2_bb'] <= y2)) & ((df_points['x1_bb'] >= x1) & ((df_points['x1_bb'] <= x2)) | ((df_points['x2_bb'] >= x1) & (df_points['x2_bb'] <= x2))))) &
                      ~((df_points['x1_bb'] >= x1) & (df_points['y1_bb'] >= y1) & (df_points['x2_bb'] <= x2) & (df_points['y2_bb'] <= y2))]

  return points_inside_block, points_in_edge_block


def calculate_area_bboxes(df, img_name):
  # calculate_area_bboxes: calculate the mean area of the objects in the image
  points = df[df['image_id'] == img_name]['geometry']

  area=[]
  for point in points:
    point = ast.literal_eval(point) # This way we have
    width = (point[1][0]-point[0][0])
    height = (point[2][1]-point[1][1])
    area.append(width*height)

  area_mean = np.mean(area)
  return area_mean



def yolo_to_rectangle(coords_yolo):
    # Function to convert from YOLO format to rectangle coordinates
    x_center, y_center, width, height = coords_yolo
    x1 = math.ceil((x_center - width / 2))
    y1 = math.ceil((y_center - height / 2))
    x2 = math.ceil((x_center + width / 2))
    y2 = math.ceil((y_center + height / 2))
    return [x1, y1, x2, y2]


############################################################################################################
#           Functions when only classifying airplanes and not the bounding boxes                           #
############################################################################################################   


def config_YOLO_file(path, name_config_file, path_data_store, name_classes):
  files_names = []

  for files in os.listdir(path):
    complete_path = os.path.join(path, files)
    if os.path.isfile(complete_path):
      files_names.append(files)

  if name_config_file in (files_names):
    return os.path.join(path, name_config_file)

  else:
    classes_txt= 'names:'
    i = 0
    for classes in name_classes:
      classes_txt += f'\n  {i}: {classes}'
      i += 1

    numb_classes = f'\nnc: {i}'

    path_txt = f'\npath: {os.path.join(path, path_data_store)}'
    train_path_txt = f'\ntrain: images/train'
    val_path_txt = f'\nval: images/train'

    CONFIG = [classes_txt, numb_classes, path_txt, train_path_txt, val_path_txt]
    print(CONFIG)
    with open(os.path.join(path, name_config_file), 'a') as f:
      for line in CONFIG:
       f.write(line)

    return os.path.join(path, name_config_file)
  


def label_img(df, img_name, indx_img, new_size, folder_dst_labels, point_inside_block, point_in_edge_block, x1_block, y1_block, x2_block, y2_block):
  # label_img has the aim of converting the BB from the dataframe to the YOLO format
  # to this aim there is the need to do some operations. such a result of cropping the images

  file_name = folder_dst_labels+ '/' + img_name[:-4]+f'_{indx_img}.txt'
  open(file_name, 'w').close()


  # Compute the min and max area of the objects in this image that will be used to those boxes detected that are small due to the subdividing images, that then, there will not be included in the YOLO file
  area_mean = calculate_area_bboxes(df, img_name)

  for i in range(0, len(point_inside_block)):
      x1_bb = point_inside_block.iloc[i]['x1_bb'] - x1_block
      y1_bb = point_inside_block.iloc[i]['y1_bb'] - y1_block
      x2_bb = point_inside_block.iloc[i]['x2_bb'] - x1_block
      y2_bb = point_inside_block.iloc[i]['y2_bb'] - y1_block

      x0 = int((x1_bb+x2_bb)/2) # tiene que ser relativo a la imagen
      y0 = int((y1_bb+y2_bb)/2) # tiene que ser relativo a la imagen

      w = int(x2_bb-x1_bb) # Tiene que ser relativo a la imagen
      h = int(y2_bb-y1_bb) # Tiene que ser relativo a la imagen

      '''if indx_img == 17:
        print('Inside')
        print(x1_bb,y1_bb,x2_bb, y2_bb)'''

      if w*h > area_mean*0.3:
        
        with open(file_name, 'r') as file:
            already_in_file = False
            n_lines = 0

            for line in file:
              n_lines +=1             
              a  = (line.split())
              if a != []:
                  _,x_read,y_read,w_read,h_read = [float(x) for x in a]
                  # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
                  # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
                  xmin_read = ((x_read - w_read / 2))*new_size
                  ymin_read = ((y_read - h_read / 2))*new_size
                  xmax_read = ((x_read + w_read / 2))*new_size
                  ymax_read = ((y_read + h_read / 2))*new_size
                  if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
                    already_in_file = True

            if not already_in_file:
              if n_lines>0:
                with open(file_name, 'a') as file:
                  txt = f'\n0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)
              else:
                with open(file_name, 'w') as file:
                  txt = f'0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)

  for i in range(0, len(point_in_edge_block)):
      x1_bb = point_in_edge_block.iloc[i]['x1_bb']
      y1_bb = point_in_edge_block.iloc[i]['y1_bb']
      x2_bb = point_in_edge_block.iloc[i]['x2_bb']
      y2_bb = point_in_edge_block.iloc[i]['y2_bb']

      if x1_bb < x1_block:
        x1_bb = x1_block +1

      if y1_bb < y1_block:
        y1_bb = y1_block +1

      if x2_bb > x2_block:
        x2_bb = x2_block -1

      if y2_bb > y2_block:
        y2_bb = y2_block -1

      x1_bb = x1_bb - x1_block
      y1_bb = y1_bb - y1_block
      x2_bb = x2_bb - x1_block
      y2_bb = y2_bb - y1_block

      x_max = math.floor(x2_block/new_size)*new_size
      y_max = math.floor(y2_block/new_size)*new_size
      # REVISAR LO DE NEW_SIZE, YA QUE AHORA EN ESTOS CASOS ALOMEJOR NO ES ENTRE 512

      x0 = int((x1_bb+x2_bb)/2) # tiene que ser relativo a la imagen
      y0 = int((y1_bb+y2_bb)/2) # tiene que ser relativo a la imagen

      if x0 > x_max:
        x0 = x_max -1

      if y0 > y_max:
        y0 = y_max -1


      w = int(x2_bb-x1_bb) # Tiene que ser relativo a la imagen
      h = int(y2_bb-y1_bb) # Tiene que ser relativo a la imagen

      '''if indx_img == 17:
        print(f'x1_block: {x1_block}, y1_block: {y1_block}, x2_block: {x2_block}, y2: {y2_bb}')

        print(f'x1_bb: {x1_bb}, y1_bb: {y1_bb}, x2_bb: {x2_bb}, y2_bb: {y2_bb}')
        print(f'x0: {x0}, y0: {y0}, w: {w}, h: {h}')'''

      if w*h > area_mean*0.3:
                
        with open(file_name, 'r') as file:
            already_in_file = False

            n_lines = 0
            
            for line in file:
              n_lines +=1  
              a  = (line.split())
              if a != []:
                  _,x_read,y_read,w_read,h_read = [float(x) for x in a]
                  # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
                  # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
                  xmin_read = ((x_read - w_read / 2))*new_size
                  ymin_read = ((y_read - h_read / 2))*new_size
                  xmax_read = ((x_read + w_read / 2))*new_size
                  ymax_read = ((y_read + h_read / 2))*new_size
                  if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
                    already_in_file = True

            if not already_in_file:
              if n_lines>0:
                with open(file_name, 'a') as file:
                  txt = f'\n0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)
              else:
                with open(file_name, 'w') as file:
                  txt = f'0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)


def YOLO_dividing_image(folder, df, new_size, folder_dst, folder_dst_labels):
  # YOLO_dividing_image: create the subdividing images and the corresponding YOLO files
  # for creating the dividing the images we need to first crop the images and then adjust the labels
  # folder: the folder where are the images we want to resize
  # df: dataframe where is store the information of the Boundary Boxes of the image
  # new_size: the new size of the crop images

  # Name of all the files that are inside the folder images
  files_names = extract_image_name(folder)

  # If the destination is not create it, make one
  os.makedirs(folder_dst, exist_ok=True)
  loop = tqdm(files_names, total = len(files_names), position = 0, leave = True)
  for img_name in loop:
    # print(img_name)

    '''1º Obtain the points and classes of the corresponding image'''
    data = []
    index = 0
    for point in df[df['image_id'] == img_name]['geometry']:
      point = ast.literal_eval(point) # This way we have
      x1_bb = point[0][0] # Boundary box x1
      y1_bb = point[0][1] # Boundary box y1
      x2_bb = point[2][0] # Boundary box x2
      y2_bb = point[2][1] # Boundary box y2
      data.append({'id': index, 'x1_bb': x1_bb, 'y1_bb': y1_bb, 'x2_bb': x2_bb, 'y2_bb': y2_bb})
      index += 1

    # Dataframe of the points obtained of this image
    df_points = pd.DataFrame(data)

    '''2º Generate the crop images'''

    # Load the image
    if img_name.endswith('.jpg'):
      img = Image.open(folder + f'/{img_name}')
    # size of the original images
    x_size, y_size = img.size

    # nblocks_x refers to the number of divisions in the x axis
    # nblocks_y refers to the number of divisions in the y axis
    nblocks_x = int(x_size/new_size)
    nblocks_y = int(y_size/new_size)


    indx_img = 1
    for i_y in range(0,nblocks_y):
      for i_x in range(0, nblocks_x):
        x1 = new_size * i_x
        y1 = new_size * i_y
        x2 = new_size * (i_x+1)
        y2 = new_size * (i_y+1)

        ################## print(f'Block: {i_x+(i_y+1)*nblocks_y}')
        '''2.1º Search for the elements that are completely inside of the block and those for are just divided due to the division of the block'''
        # Que todos los puntos esten dentro del block
        filtered_points_inside_block, filtered_points_in_edge_block= filter_points(df_points, x1, y1, x2, y2)

        '''2.2º Genererate the crop for those blocks that are inside of this block'''
        # Generamos el crop de este block
        img_crop = crop_image(folder, img_name, x1, y1, x2, y2)
        img_crop.save(folder_dst+'/'+img_name[:-4]+f'_{indx_img}.jpg')
        ########################### print(img_name[:-4]+f'_{indx_img}.jpg')

        '''2.3º Create the corresponging YOLO label file for the image generated'''
        label_img(df, img_name, indx_img, new_size,folder_dst_labels, filtered_points_inside_block, filtered_points_in_edge_block, x1, y1, x2, y2)
        indx_img = indx_img + 1


        '''2.5º Generate a crop for those boundary boxes that are cut due to the division of the blocks'''
        ''' to this aim son adjusts must be done'''
        if not filtered_points_in_edge_block.empty:
          while not filtered_points_in_edge_block.empty:
            id = filtered_points_in_edge_block.iloc[0]['id']
            x1_bb = filtered_points_in_edge_block.iloc[0]['x1_bb']
            y1_bb = filtered_points_in_edge_block.iloc[0]['y1_bb']
            x2_bb = filtered_points_in_edge_block.iloc[0]['x2_bb']
            y2_bb = filtered_points_in_edge_block.iloc[0]['y2_bb']

            # Ajsutamos la división de las imagenes para intentar generar más imagenes en las que no se encuentra un avión a la mitad
            x1_filter, y1_filter, x2_filter, y2_filter = x1, y1, x2, y2
            x1_diff = 0
            y1_diff = 0
            x2_diff = 0
            y2_diff = 0
            if x1_bb < x1_filter:
              x1_diff = x1_filter - x1_bb
              x1_filter = x1_bb;
              x2_filter = x1_filter+new_size
            if y1_bb < y1_filter:
              y1_diff = y1_filter - y1_bb
              y1_filter = y1_bb
              y2_filter = y1_filter+new_size
            if x2_bb > x2_filter:
              x2_diff =  x2_bb - x2_filter
              x2_filter = x2_bb
              x1_filter = x2_filter - new_size
            if y2_bb > y2_filter:
              y2_diff =  y2_bb - y2_filter
              y2_filter = y2_bb
              y1_filter = y2_filter- new_size



            # Generamos el crop de este block
            img_crop = crop_image(folder, img_name, x1_filter, y1_filter, x2_filter, y2_filter)
            img_crop.save(folder_dst+'/'+img_name[:-4]+f'_{indx_img}.jpg')

            ########### print(img_name[:-4]+f'_{indx_img}.jpg   ____')

            # Eliminamos la fila que ya hemos utilizado
            filtered_points_in_edge_block = filtered_points_in_edge_block.drop(filtered_points_in_edge_block[filtered_points_in_edge_block['id']==id].index)


            # Ahora buscamos en la nueva lista, a ver si hay algun avión de los incluidos
            #en la lista filtered_points_in_edge_block tambien se ha incluido en esta imagen, y por tanto no hay que crearle una particular a esta
            indices = filtered_points_in_edge_block[(filtered_points_in_edge_block['x1_bb'] >= x1_filter) & (filtered_points_in_edge_block['y1_bb'] >= y1_filter) & (filtered_points_in_edge_block['x2_bb'] <= x2_filter) & (filtered_points_in_edge_block['y2_bb'] <= y2_filter)].index

            # Eliminamos la fila de los objetos que debido a este nuevo crop ya estan incluidas en esta nueva imagen
            filtered_points_in_edge_block = filtered_points_in_edge_block.drop(indices)

            df2_points = df_points.copy()
            df2_points['x1_bb'] = df2_points['x1_bb']
            df2_points['y1_bb'] = df2_points['y1_bb']
            df2_points['x2_bb'] = df2_points['x2_bb']
            df2_points['y2_bb'] = df2_points['y2_bb']


            '''2.6.1º Search for the elements that are completely inside of the block and those for are just divided due to the division of the block'''
            # Que todos los puntos esten dentro del block
            filtered_points_inside_block_crop, filtered_points_in_edge_block_crop= filter_points(df2_points, x1_filter, y1_filter, x2_filter, y2_filter)

            '''2.6º Over the new crop image gennerate we need to repeat the process for the labelling'''
            # df2_points = df_points.copy()
            # df2_points['x1_bb'] = df2_points['x1_bb'] #+ x1_diff - x2_diff
            # df2_points['y1_bb'] = df2_points['y1_bb'] #+ y1_diff - y2_diff
            # df2_points['x2_bb'] = df2_points['x2_bb'] #+ x1_diff - x2_diff
            # df2_points['y2_bb'] = df2_points['y2_bb'] #+ y1_diff - y2_diff

            '''if indx_img == 17:
              print(filtered_points_inside_block)
              print(filtered_points_in_edge_block_crop)
              print(f'x1_diff: {x1_diff}, y1_diff: {y1_diff}, x2_diff: {x2_diff}, y2_diff: {y2_diff}')'''

            '''2.6.1º Search for the elements that are completely inside of the block and those for are just divided due to the division of the block'''
            # Que todos los puntos esten dentro del block
            filtered_points_inside_block_crop, filtered_points_in_edge_block_crop= filter_points(df2_points, x1_filter, y1_filter, x2_filter, y2_filter)


            '''2.6.2º Create the corresponging YOLO label file for the image generated'''
            label_img(df, img_name, indx_img, new_size,folder_dst_labels, filtered_points_inside_block_crop, filtered_points_in_edge_block_crop, x1_filter, y1_filter, x2_filter, y2_filter)
            indx_img = indx_img + 1



def label_img_val_test(df, folder, img_name, folder_dst_labels):
  # label_img_val_test: create the YOLO file for the validation and test images
  # this is different from the previous label function as in this we dont need to do any crop
  file_name = folder_dst_labels+ '/' + img_name[:-4]+'.txt'
  open(file_name, 'w').close()
  print(file_name)

  for point in df[df['image_id'] == img_name]['geometry']:
    point = ast.literal_eval(point) # This way we have
    x1_bb = point[0][0] # Boundary box x1
    y1_bb = point[0][1] # Boundary box y1
    x2_bb = point[2][0] # Boundary box x2
    y2_bb = point[2][1] # Boundary box y2

    x0 = (x1_bb+x2_bb)/2
    y0 = (y1_bb+y2_bb)/2
    w = (x2_bb-x1_bb) # Tiene que ser relativo a la imagen
    h = (y2_bb-y1_bb) # Tiene que ser relativo a la imagen

    # Load the image
    img = Image.open(folder + f'/{img_name}')
    # size of the original images
    x_size, y_size = img.size

    
    with open(file_name, 'r') as file:
      already_in_file = False
      n_lines = 0
      
      for line in file:
        n_lines +=1    
        a  = (line.split())
        if a != []:
            _,x_read,y_read,w_read,h_read = [float(x) for x in a]
            # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
            # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
            xmin_read = ((x_read - w_read / 2))*x_size
            ymin_read = ((y_read - h_read / 2))*y_size
            xmax_read = ((x_read + w_read / 2))*x_size
            ymax_read = ((y_read + h_read / 2))*y_size
            if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
              already_in_file = True


      if not already_in_file:
        if n_lines>0:
          with open(file_name, 'a') as file:
            txt = f'\n0 {x0/x_size:.6f} {y0/y_size:.6f} {w/x_size:.6f} {h/y_size:.6f}'
            file.write(txt)
        else:
          with open(file_name, 'w') as file:
            txt = f'0 {x0/x_size:.6f} {y0/y_size:.6f} {w/x_size:.6f} {h/y_size:.6f}'
            file.write(txt)     



#########################################################################
#               Functions for Truncated Airplanes                       #            
#########################################################################
def config_YOLO_file_truncated_airplane(path, name_config_file, path_data_store, name_classes):
  # config_YOLO_file_truncated_airplane: create the YOLO configuration file for the truncated airplanes
  files_names = []

  for files in os.listdir(path):
    complete_path = os.path.join(path, files)
    if os.path.isfile(complete_path):
      files_names.append(files)

  if name_config_file in (files_names):
    return os.path.join(path, name_config_file)

  else:
    classes_txt= 'names:'
    i = 0
    for classes in name_classes:
      classes_txt += f'\n  {i}: {classes}'
      i += 1

    numb_classes = f'\nnc: {i}'

    path_txt = f'\npath: {os.path.join(path, path_data_store)}'
    train_path_txt = f'\ntrain: images/train'
    val_path_txt = f'\nval: images/train'

    CONFIG = [classes_txt, numb_classes, path_txt, train_path_txt, val_path_txt]
    print(CONFIG)
    with open(os.path.join(path, name_config_file), 'a') as f:
      for line in CONFIG:
       f.write(line)

    return os.path.join(path, name_config_file)



def label_img_truncated_airplane(df, img_name, indx_img, new_size, folder_dst_labels, point_inside_block, point_in_edge_block, x1_block, y1_block, x2_block, y2_block):
  file_name = folder_dst_labels+ '/' + img_name[:-4]+f'_{indx_img}.txt'
  open(file_name, 'w').close()


  # Compute the min and max area of the objects in this image that will be used to those boxes detected that are small due to the subdividing images, that then, there will not be included in the YOLO file
  area_mean = calculate_area_bboxes(df, img_name)

  for i in range(0, len(point_inside_block)):
      x1_bb = point_inside_block.iloc[i]['x1_bb'] - x1_block
      y1_bb = point_inside_block.iloc[i]['y1_bb'] - y1_block
      x2_bb = point_inside_block.iloc[i]['x2_bb'] - x1_block
      y2_bb = point_inside_block.iloc[i]['y2_bb'] - y1_block

      x0 = int((x1_bb+x2_bb)/2) # tiene que ser relativo a la imagen
      y0 = int((y1_bb+y2_bb)/2) # tiene que ser relativo a la imagen

      w = int(x2_bb-x1_bb) # Tiene que ser relativo a la imagen
      h = int(y2_bb-y1_bb) # Tiene que ser relativo a la imagen


      if w*h > area_mean*0.3:
                               

        with open(file_name, 'r') as file:
            already_in_file = False
            n_lines = 0
            
            for line in file:
              n_lines +=1    
              a  = (line.split())
              if a != []:
                  _,x_read,y_read,w_read,h_read = [float(x) for x in a]
                  # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
                  # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
                  xmin_read = ((x_read - w_read / 2))*new_size
                  ymin_read = ((y_read - h_read / 2))*new_size
                  xmax_read = ((x_read + w_read / 2))*new_size
                  ymax_read = ((y_read + h_read / 2))*new_size
                  if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
                    already_in_file = True

            if not already_in_file:
              if n_lines>0:
                with open(file_name, 'a') as file:
                  if point_inside_block.iloc[i]['class_detected'].item() == 'Airplane':
                    txt = f'\n0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  elif point_inside_block.iloc[i]['class_detected'].item() == 'Truncated_airplane':
                    txt = f'\n1 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)
              else:
                with open(file_name, 'w') as file:
                  if point_inside_block.iloc[i]['class_detected'].item() == 'Airplane':
                    txt = f'0 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  elif point_inside_block.iloc[i]['class_detected'].item() == 'Truncated_airplane':
                    txt = f'1 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'                  
                  file.write(txt)

  for i in range(0, len(point_in_edge_block)):
      x1_bb = point_in_edge_block.iloc[i]['x1_bb']
      y1_bb = point_in_edge_block.iloc[i]['y1_bb']
      x2_bb = point_in_edge_block.iloc[i]['x2_bb']
      y2_bb = point_in_edge_block.iloc[i]['y2_bb']

      if x1_bb < x1_block:
        x1_bb = x1_block +1

      if y1_bb < y1_block:
        y1_bb = y1_block +1

      if x2_bb > x2_block:
        x2_bb = x2_block -1

      if y2_bb > y2_block:
        y2_bb = y2_block -1

      x1_bb = x1_bb - x1_block
      y1_bb = y1_bb - y1_block
      x2_bb = x2_bb - x1_block
      y2_bb = y2_bb - y1_block

      x_max = math.floor(x2_block/new_size)*new_size
      y_max = math.floor(y2_block/new_size)*new_size
      # REVISAR LO DE NEW_SIZE, YA QUE AHORA EN ESTOS CASOS ALOMEJOR NO ES ENTRE 512

      x0 = int((x1_bb+x2_bb)/2) # tiene que ser relativo a la imagen
      y0 = int((y1_bb+y2_bb)/2) # tiene que ser relativo a la imagen

      if x0 > x_max:
        x0 = x_max -1

      if y0 > y_max:
        y0 = y_max -1


      w = int(x2_bb-x1_bb) # Tiene que ser relativo a la imagen
      h = int(y2_bb-y1_bb) # Tiene que ser relativo a la imagen

      if w*h > area_mean*0.3:
        
        with open(file_name, 'r') as file:
            already_in_file = False
            n_lines = 0
            
            for line in file:
              n_lines +=1          
              a  = (line.split())
              if a != []:
                  _,x_read,y_read,w_read,h_read = [float(x) for x in a]
                  # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
                  # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
                  xmin_read = ((x_read - w_read / 2))*new_size
                  ymin_read = ((y_read - h_read / 2))*new_size
                  xmax_read = ((x_read + w_read / 2))*new_size
                  ymax_read = ((y_read + h_read / 2))*new_size
                  if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
                    already_in_file = True

            if not already_in_file:
              if n_lines>0:
                with open(file_name, 'a') as file:
                  txt = f'\n1 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)
              else:
                with open(file_name, 'w') as file:
                  txt = f'1 {x0/new_size:.6f} {y0/new_size:.6f} {w/new_size:.6f} {h/new_size:.6f}'
                  file.write(txt)



def YOLO_dividing_image_truncated_airplane(folder, df, new_size, folder_dst, folder_dst_labels):
  # YOLO_dividing_image: create the subdividing images and the corresponding YOLO files
  # folder: the folder where are the images we want to resize
  # df: dataframe where is store the information of the Boundary Boxes of the image
  # new_size: the new size of the crop images

  # Name of all the files that are inside the folder images
  files_names = extract_image_name(folder)

  # If the destination is not create it, make one
  os.makedirs(folder_dst, exist_ok=True)

  for img_name in files_names:
    print(img_name)

    '''1º Obtain the points and classes of the corresponding image'''
    points = []

    data = []
    index = 0
    for point in df[df['image_id'] == img_name]['geometry']:
      class_detected = df[(df['image_id'] == img_name) & (df['geometry'] == point)]['class']
      point = ast.literal_eval(point) # This way we have
      x1_bb = point[0][0] # Boundary box x1
      y1_bb = point[0][1] # Boundary box y1
      x2_bb = point[2][0] # Boundary box x2
      y2_bb = point[2][1] # Boundary box y2
      data.append({'id': index, 'x1_bb': x1_bb, 'y1_bb': y1_bb, 'x2_bb': x2_bb, 'y2_bb': y2_bb, 'class_detected': class_detected})
      index += 1

    # Dataframe of the points obtained of this image
    df_points = pd.DataFrame(data)

    '''2º Generate the crop images'''

    # Load the image
    img = Image.open(folder + f'/{img_name}')
    # size of the original images
    x_size, y_size = img.size

    # nblocks_x refers to the number of divisions in the x axis
    # nblocks_y refers to the number of divisions in the y axis
    nblocks_x = int(x_size/new_size)
    nblocks_y = int(y_size/new_size)


    indx_img = 1
    for i_y in range(0,nblocks_y):
      for i_x in range(0, nblocks_x):
        x1 = new_size * i_x
        y1 = new_size * i_y
        x2 = new_size * (i_x+1)
        y2 = new_size * (i_y+1)

        ################## print(f'Block: {i_x+(i_y+1)*nblocks_y}')
        '''2.1º Search for the elements that are completely inside of the block and those for are just divided due to the division of the block'''
        # Que todos los puntos esten dentro del block
        filtered_points_inside_block, filtered_points_in_edge_block= filter_points(df_points, x1, y1, x2, y2)
  
        '''2.2º Genererate the crop for those blocks that are inside of this block'''
        # Generamos el crop de este block
        img_crop = crop_image(folder, img_name, x1, y1, x2, y2)
        img_crop.save(folder_dst+'/'+img_name[:-4]+f'_{indx_img}.jpg')
        ########################### print(img_name[:-4]+f'_{indx_img}.jpg')

        '''2.3º Create the corresponging YOLO label file for the image generated'''
        label_img(df, img_name, indx_img, new_size,folder_dst_labels, filtered_points_inside_block, filtered_points_in_edge_block, x1, y1, x2, y2)
        indx_img = indx_img + 1


        '''2.5º Generate a crop for those boundary boxes that are cut due to the division of the blocks'''
        ''' to this aim son adjusts must be done'''
        if not filtered_points_in_edge_block.empty:
          while not filtered_points_in_edge_block.empty:
            id = filtered_points_in_edge_block.iloc[0]['id']
            x1_bb = filtered_points_in_edge_block.iloc[0]['x1_bb']
            y1_bb = filtered_points_in_edge_block.iloc[0]['y1_bb']
            x2_bb = filtered_points_in_edge_block.iloc[0]['x2_bb']
            y2_bb = filtered_points_in_edge_block.iloc[0]['y2_bb']

            # Ajsutamos la división de las imagenes para intentar generar más imagenes en las que no se encuentra un avión a la mitad
            x1_filter, y1_filter, x2_filter, y2_filter = x1, y1, x2, y2
            x1_diff = 0
            y1_diff = 0
            x2_diff = 0
            y2_diff = 0
            if x1_bb < x1_filter:
              x1_diff = x1_filter - x1_bb
              x1_filter = x1_bb;
              x2_filter = x1_filter+new_size
            if y1_bb < y1_filter:
              y1_diff = y1_filter - y1_bb
              y1_filter = y1_bb
              y2_filter = y1_filter+new_size
            if x2_bb > x2_filter:
              x2_diff =  x2_bb - x2_filter
              x2_filter = x2_bb
              x1_filter = x2_filter - new_size
            if y2_bb > y2_filter:
              y2_diff =  y2_bb - y2_filter
              y2_filter = y2_bb
              y1_filter = y2_filter- new_size



            # Generamos el crop de este block
            img_crop = crop_image(folder, img_name, x1_filter, y1_filter, x2_filter, y2_filter)
            img_crop.save(folder_dst+'/'+img_name[:-4]+f'_{indx_img}.jpg')

            ########### print(img_name[:-4]+f'_{indx_img}.jpg   ____')

            # Eliminamos la fila que ya hemos utilizado
            filtered_points_in_edge_block = filtered_points_in_edge_block.drop(filtered_points_in_edge_block[filtered_points_in_edge_block['id']==id].index)


            # Ahora buscamos en la nueva lista, a ver si hay algun avión de los incluidos
            #en la lista filtered_points_in_edge_block tambien se ha incluido en esta imagen, y por tanto no hay que crearle una particular a esta
            indices = filtered_points_in_edge_block[(filtered_points_in_edge_block['x1_bb'] >= x1_filter) & (filtered_points_in_edge_block['y1_bb'] >= y1_filter) & (filtered_points_in_edge_block['x2_bb'] <= x2_filter) & (filtered_points_in_edge_block['y2_bb'] <= y2_filter)].index

            # Eliminamos la fila de los objetos que debido a este nuevo crop ya estan incluidas en esta nueva imagen
            filtered_points_in_edge_block = filtered_points_in_edge_block.drop(indices)

            df2_points = df_points.copy()
            df2_points['x1_bb'] = df2_points['x1_bb']
            df2_points['y1_bb'] = df2_points['y1_bb']
            df2_points['x2_bb'] = df2_points['x2_bb']
            df2_points['y2_bb'] = df2_points['y2_bb']


            '''2.6.1º Search for the elements that are completely inside of the block and those for are just divided due to the division of the block'''
            # Que todos los puntos esten dentro del block
            filtered_points_inside_block_crop, filtered_points_in_edge_block_crop= filter_points(df2_points, x1_filter, y1_filter, x2_filter, y2_filter)


            '''2.6.2º Create the corresponging YOLO label file for the image generated'''
            label_img(df, img_name, indx_img, new_size,folder_dst_labels, filtered_points_inside_block_crop, filtered_points_in_edge_block_crop, x1_filter, y1_filter, x2_filter, y2_filter)
            indx_img = indx_img + 1


def label_img_val_test_truncated_airplane(df, folder, img_name, folder_dst_labels):
  file_name = folder_dst_labels+ '/' + img_name[:-4]+'.txt'
  open(file_name, 'w').close()
  print(file_name)

  for point in df[df['image_id'] == img_name]['geometry']:
    class_detected = df[(df['image_id'] == img_name) & (df['geometry'] == point)]['class']
    point = ast.literal_eval(point) # This way we have
    x1_bb = point[0][0] # Boundary box x1
    y1_bb = point[0][1] # Boundary box y1
    x2_bb = point[2][0] # Boundary box x2
    y2_bb = point[2][1] # Boundary box y2

    x0 = (x1_bb+x2_bb)/2
    y0 = (y1_bb+y2_bb)/2
    w = (x2_bb-x1_bb) # Tiene que ser relativo a la imagen
    h = (y2_bb-y1_bb) # Tiene que ser relativo a la imagen

    # Load the image
    img = Image.open(folder + f'/{img_name}')
    # size of the original images
    x_size, y_size = img.size

    if class_detected.item() == 'Airplane':
        txt = f'\n0 {x0/x_size:.6f} {y0/y_size:.6f} {w/x_size:.6f} {h/y_size:.6f}'
    elif class_detected.item() == 'Truncated_airplane':
        txt = f'\n1 {x0/x_size:.6f} {y0/y_size:.6f} {w/x_size:.6f} {h/y_size:.6f}'

    with open(file_name, 'r') as file:
      already_in_file = False
      n_lines = 0
            
      for line in file:
        n_lines +=1  
        a  = (line.split())
        if a != []:
            _,x_read,y_read,w_read,h_read = [float(x) for x in a]
            # Queremos evitar tener dos cuadrados similares, esto es debido a que en el dataframe hay cuadrados que no es que esten repetidos, si no que tienen pixeles muy similares haciendo
            # referencia al mismo avión, por tanto, vamos a decir que si hay pixeles muy similares, 'hay colision' de cuadrados y no puede haber dos aviones en el mismo cuadrado
            xmin_read = ((x_read - w_read / 2))*x_size
            ymin_read = ((y_read - h_read / 2))*y_size
            xmax_read = ((x_read + w_read / 2))*x_size
            ymax_read = ((y_read + h_read / 2))*y_size
            if (xmin_read<x0<xmax_read) and(ymin_read<y0<ymax_read):
              already_in_file = True


      if not already_in_file:
        if n_lines>0:
          with open(file_name, 'a') as file:
            file.write(txt)
        else:
          with open(file_name, 'w') as file:
            file.write(txt)

def delete_empty_txt_and_corresponding_images(txt_folder, image_folder):
    """
    Deletes .txt files with an empty first line from a specified folder and also deletes
    corresponding .jpg files in another specified folder.

    Parameters:
    - txt_folder (str): The path to the folder containing the .txt files.
    - image_folder (str): The path to the folder containing the .jpg files.
    """
    
    # Iterate over all files in the txt folder
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):  # Check if the file is a .txt file
            txt_file_path = os.path.join(txt_folder, filename)  # Full path to the txt file
            
            # Open the txt file and read the first line
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline()
                
                # Check if the first line is empty
                if first_line == '\n':
                    # Delete the txt file
                    os.remove(txt_file_path)
                    print(f"Deleted empty .txt file: {filename}")
                    
                    # Construct the corresponding image file name and path
                    image_filename = f"{os.path.splitext(filename)[0]}.jpg"
                    image_file_path = os.path.join(image_folder, image_filename)
                    
                    # Check if the corresponding image file exists and delete it
                    if os.path.exists(image_file_path):
                        os.remove(image_file_path)
                        print(f"Deleted corresponding image file: {image_filename}")