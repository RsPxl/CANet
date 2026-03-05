import random
import pandas as pd
import os
from read_gdal import GRID
import glob

os.environ["CUDA_VISIBLE_DEVICES"]="0"


size = 128
# randm sample
image_out = 'Data/DataSet_BandCombine_Train/Img_SampleSet'
label_out = 'Data/DataSet_BandCombine_Train/Lab_SampleSet'
def generate_train_dataset(image_num = 10,     #200000
                           train_image_path = image_out,
                           train_label_path = label_out):


    g_count = 0

    images_path = 'Data/DataSet_BandCombine'
    labels_path = 'Data/DataSet_BandCombine_Label'
    images_list = sorted(glob.glob(images_path+'/*.tif'))
    labels_list = sorted(glob.glob(labels_path+'/*.tif'))
    image_each = image_num
    image_path, label_path = [], []
    for i in range(len(images_list)):
        count = 0
        print(images_list[i].split('/')[-1])
        image = GRID().read_tif(images_list[i])[:,:,:7]
        label = GRID().read_tif(labels_list[i])
        
        X_height, X_width = image.shape[0], image.shape[1]
        
        print(image.shape)
        
        while count < image_each:

            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]


            image_d, label_d = image_ogi, label_ogi
            image_path.append(train_image_path+'/'+'%05d.tif' % g_count)
            label_path.append(train_label_path+'/'+'%05d.tif' % g_count)
            
            GRID().write_tif((train_image_path+'/'+'%05d.tif' % g_count), image_d)
            GRID().write_tif((train_label_path+'/'+'%05d.tif' % g_count), label_d)
            print(g_count)
            count += 1
            g_count += 1
    df = pd.DataFrame({'image':image_path, 'label':label_path})
    df.to_csv('Data/train_path_list.csv', index=False)


if __name__ == '__main__':
    if not os.path.exists('Data/DataSet_BandCombine_Train/Img_SampleSet'): os.mkdir('Data/DataSet_BandCombine_Train/Img_SampleSet')
    if not os.path.exists('Data/DataSet_BandCombine_Train/Lab_SampleSet'): os.mkdir('Data/DataSet_BandCombine_Train/Lab_SampleSet')
    generate_train_dataset()
