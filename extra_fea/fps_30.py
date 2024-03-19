import os
import glob
import subprocess



video_dir_path = r'D:\Desktop\code\6th-ABAW\test_data\raw'

videos_path = glob.glob(os.path.join(video_dir_path, '*'))
videos_path.sort()

# videos_path = videos_path[1:]

# print(videos_path[:3])
# len(videos_path)

video = 'video'
save_path = os.path.join(r'D:\Desktop\code\6th-ABAW\test_data',video)
if not os.path.exists(save_path): os.makedirs(save_path)



for video_path in videos_path:
    filename = os.path.basename(video_path)[:-4]
    print(filename)
    output_path = os.path.join(save_path, filename) 
    
    output_path = f"{output_path}.mp4"
    # print(output_path)y

    command = f"ffmpeg -i {video_path} -r 30 {output_path}"
    subprocess.run(command, shell=True)