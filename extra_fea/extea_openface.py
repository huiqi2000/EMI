import os
import glob
import pandas as pd
import numpy as np
import cv2
import pickle


def read_csv(csv_path):
    df = pd.read_csv(csv_path)

    data1 = df.iloc[:,-35:-2]
    data2 = df.iloc[:, -1:]
    data = np.concatenate((data1.to_numpy(), data2.to_numpy()), axis=1)

    return data


def generate_face_videoOne(input_root, save_root):
    for dir_path in glob.glob(input_root + '/*_aligned'): # 'xx/xx/000100_guest_aligned'
        frame_names = os.listdir(dir_path) # ['xxx.bmp']
        for ii in range(len(frame_names)):
            frame_path = os.path.join(dir_path, frame_names[ii]) # 'xx/xx/000100_guest_aligned/xxx.bmp'
            bmp_img = cv2.imread(frame_path)
            cv2.imwrite(f'{save_root}/{ii}.jpg', bmp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            


def generate_aus(input_root, save_root):
    for csv_path in glob.glob(input_root + '/*.csv'):
        csv_name = os.path.basename(csv_path)[:-4]

        feature = read_csv(csv_path)
        save_path = os.path.join(save_root, csv_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(feature, f)

def extract(input_dir, save_dir, face_dir, aus_dir):

    print(f'Find total "{len(input_dir)}" videos.')
    
    for i in range(len(input_dir)):
        video_path = input_dir[i]
        filename = os.path.basename(video_path)[:-4]


        input_root = video_path # exists
        save_root  = os.path.join(save_dir, filename)
        face_root  = os.path.join(face_dir, filename)
        aus_root = aus_dir
        
        
        if not os.path.exists(save_root): os.makedirs(save_root)
        if not os.path.exists(face_root): os.makedirs(face_root)
        # if not os.path.exists(aus_root): os.makedirs(aus_root)
        
        
        exe_path = os.path.join(r'D:\Desktop\code\bishe\pre-data\video\OpenFace_2.2.0_win_x64', 'FeatureExtraction.exe')
        commond = '%s -f \"%s\" -out_dir \"%s\"' % (exe_path, input_root, save_root)
        os.system(commond)
        generate_face_videoOne(save_root, face_root)
        generate_aus(save_root,aus_root )


if __name__ == '__main__':
    
    print(f'==> Extracting openface features...')

    
    

    dataset = r'D:\Desktop\code\6th-ABAW\test_data'
    openface_all = 'openface_all'
    openface_img = 'openface_img'
    openface_aus = 'openface_aus'

    #输入视频路径路径
    video = 'video'
    video_path = os.path.join(dataset, video)
    videos_path = glob.glob(os.path.join(video_path, '*'))
    videos_path.sort(key=lambda x: int(os.path.basename(x)[:-4]))

    # 输出路径
    save_dir = os.path.join(dataset, openface_all)
    face_dir = os.path.join(dataset, openface_img)
    aus_dir = os.path.join(dataset, openface_aus)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    
    if not os.path.exists(aus_dir):
        os.makedirs(aus_dir)
    

    # process
    extract(videos_path, save_dir, face_dir, aus_dir)

    print(f'==> Finish')