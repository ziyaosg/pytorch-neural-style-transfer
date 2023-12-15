import os
import cv2 
import glob
import natsort
import argparse


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
        

def frame_putback(frames_dir, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_name)
    
    # define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = 1080, 1920
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 24.0, (frame_width, frame_height))

    extension = '*.jpg'
    frame_paths = glob.glob(os.path.join(frames_dir, extension))
    frame_paths = natsort.natsorted(frame_paths, reverse=True)


    for frame_path in frame_paths:
        
        frame = cv2.imread(frame_path)
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        video_writer.write(frame_resized)

    # release when job is finished
    video_writer.release()


# Driver Code 
if __name__ == '__main__': 

    # fixed args
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    # for videos
    content_video_dir = os.path.join(default_resource_dir, 'content-videos')
    default_frame_dir = os.path.join(default_resource_dir, 'content-frames')
    default_transferred_dir = os.path.join(default_resource_dir, 'transferred-videos')
    # for images
    # content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    
    # modifiable args
    parser = argparse.ArgumentParser()
    # for videos
    parser.add_argument("--content_video_name", type=str, help="content video name", default='test.mov')
    parser.add_argument("--frame_dir", type=str, help="output video frames directory", default=default_frame_dir)
    parser.add_argument("--transferred_video_dir", type=str, help="transferred video output directory", default=default_transferred_dir)
    args = parser.parse_args()
    

    # putting all args into config dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    # for video
    config['content_video_path'] = os.path.join(content_video_dir, config['content_video_name'])
    

    # helper functions
    frame_capture(config['content_video_path'], config['frame_dir']) 
    frame_putback(config['frame_dir'], config['transferred_video_dir'], 'output_video.mp4')
