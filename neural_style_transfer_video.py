import os
import cv2 
import glob
import shutil
import natsort
import argparse
from neural_style_transfer_blend import neural_style_transfer


FRMAES_PER_SEC = 24
FRAME_WIDTH = 360
FRAME_HEIGHT = 640
ITERATIONS = 700


def frame_capture(input_path, output_dir): 
    os.makedirs(output_dir, exist_ok=True)
    vidObj = cv2.VideoCapture(input_path) 
    count = 0

    while vidObj.isOpened(): 
        success, image = vidObj.read()
        if not success:
            break
        cv2.imwrite(os.path.join(output_dir, "frame%d.jpg" % count), image) 
        count += 1
    
    vidObj.release()
    

def frame_putback(frames_dir, video_dir, video_name):
    os.makedirs(video_dir, exist_ok=True)
    if not video_name.lower().endswith('mp4'):
        video_name += '.mp4'
    output_video_path = os.path.join(video_dir, video_name)
    
    # define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FRMAES_PER_SEC, (FRAME_WIDTH, FRAME_HEIGHT))

    extension = '*.jpg'
    frame_paths = glob.glob(os.path.join(frames_dir, extension))
    frame_paths = natsort.natsorted(frame_paths)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        video_writer.write(frame_resized)

    # release when job is finished
    video_writer.release()


def video_transfer(config):
    os.makedirs(config['transferred_frames_dir'], exist_ok=True) 

    count = 0
    extension = '*.jpg'
    frame_paths = glob.glob(os.path.join(config['content_frame_dir'], extension))
    frame_paths = natsort.natsorted(frame_paths)
    for frame_path in frame_paths:
        config['content_img_name'] = os.path.split(frame_path)[-1]

        print("now transferring frame: ", config['content_img_name'])
        dump_path = neural_style_transfer(config, lbfgs_iterations=ITERATIONS)


        copy_source = os.path.join(dump_path, config['transferred_frame_name'] + config['img_format'][1])
        copy_dest = os.path.join(config['transferred_frames_dir'], config['transferred_frame_name'] + '_' + str(count) + config['img_format'][1])
        shutil.copyfile(copy_source, copy_dest)
        video_name = os.path.split(dump_path)[-1]
        count += 1
    return video_name


if __name__ == '__main__': 

    # fixed args
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    # for videos
    content_video_dir = os.path.join(default_resource_dir, 'content-videos')
    default_frame_dir = os.path.join(default_resource_dir, 'content-frames')
    default_trans_frame_dir = os.path.join(default_resource_dir, 'transferred-frames')
    default_trans_vid_dir = os.path.join(default_resource_dir, 'transferred-videos')
    # for images
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    
    # modifiable args
    parser = argparse.ArgumentParser()
    # for videos
    parser.add_argument("--content_video_name", type=str, help="content video name", default='test.mov')
    #for images
    parser.add_argument("--style_img_name", type=str, help="style image name", default='miscellaneous_1.jpg,miscellaneous_3.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=500)

    parser.add_argument("--style_blend_weights", default=None)
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=5e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19', 'resnet'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--init_style_index", type=int, help="choosing which style image as initialization (index starts with 1)", default=1)
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=50)
    args = parser.parse_args()
    

    # putting all args into config dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    # for video
    config['content_video_path'] = os.path.join(content_video_dir, config['content_video_name'])
    config['content_frame_dir'] = default_frame_dir
    config['transferred_frames_dir'] = default_trans_frame_dir
    config['transferred_video_dir'] = default_trans_vid_dir
    config['transferred_frame_name'] = config['model'] + '_' + str(ITERATIONS).zfill(img_format[0])
    # for images
    config['content_images_dir'] = config['content_frame_dir']
    config['style_images_dir'] = style_images_dir
    config['output_img_dir'] = output_img_dir
    config['img_format'] = img_format
    

    # helper functions
    frame_capture(config['content_video_path'], config['content_frame_dir']) 
    video_name = video_transfer(config)
    frame_putback(config['transferred_frames_dir'], config['transferred_video_dir'], video_name)
