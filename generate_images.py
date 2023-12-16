import argparse
import os

from neural_style_transfer_blend import neural_style_transfer

NUM_ITERATIONS = 400

def gather_content_images(content_path):
    images = []

    for filename in os.listdir(content_path):
        filepath = os.path.join(content_path, filename)

        # Check if the file is a regular file and has a valid image extension (e.g., '.jpg', '.png')
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
            images.append(filename)

    return images

def gather_styles(styles_path: str, style_prefixes = ['picasso', 'pollock', 'van-gogh', 'miscellaneous']):
    """
    Given a directory path to the styles' directory and a list of style prefixes, collect
    all images from that directory which correspond to the given styles
    @param styles_path: path to the styles directory
    @param style_prefixes: list of style prefixes.
    @return: dictionary with keys being style prefixes and values being a list of image file names.
        For example, one entry in the returned dictionary might be 'pollck': ['pollock_autumn-rhythm.jpg', 'pollock_croaking-movement.jpg']
    """

    style_images = dict()

    # Validate the styles_path is a directory
    if not os.path.isdir(styles_path):
        raise ValueError(f"'{styles_path}' is not a valid directory path.")

    for prefix in style_prefixes:
        style_images[prefix] = []

    for filename in os.listdir(styles_path):
        filepath = os.path.join(styles_path, filename)

        # Check if the file is a regular file and has a valid image extension (e.g., '.jpg', '.png')
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
            for prefix in style_prefixes:
                # Check if the filename starts with the specified prefix
                if filename.lower().startswith(prefix.lower()):
                    style_images[prefix].append(filename)

    return style_images

def set_weights(config, weights):
    config['content_weights'] = weights[0]
    config['style_weights'] = weights[1]
    config['tv_weights'] = weights[2]

if __name__ == '__main__':
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images', 'blended')
    plots_dir = os.path.join(default_resource_dir, 'plots')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='vg_starry_night.jpg')
    parser.add_argument("--style_blend_weights", default=None)
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19', 'resnet'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--init_style_index", type=int,
                        help="choosing which style image as initialization (index starts with 1)", default=1)
    parser.add_argument("--saving_freq", type=int,
                        help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    # some values of weights that worked for figures.jpg, vg_starry_night.jpg (starting point for finding good images)
    # once you understand what each one does it gets really easy -> also see README.md

    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format
    optimization_config['plots_dir'] = plots_dir

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)

    # results_path = neural_style_transfer(optimization_config)
    style_images = gather_styles(style_images_dir)
    content_images = gather_content_images(content_images_dir)
    print(f"Styles: {style_images}")
    print(f"Contents: {content_images}")
    optimization_config['saving_freq'] = -1
    for content in content_images:
        optimization_config['content_img_name'] = content
        for style, image_names in style_images.items():
            optimization_config['style_img_name'] = ','.join(image_names)
            # VGGs
            set_weights(optimization_config, (1e5, 1e4, 1e0))
            optimization_config['model'] = 'vgg16'
            print(f"content = {content}, style = {style}, model = {optimization_config['model']}")
            neural_style_transfer(optimization_config, NUM_ITERATIONS)
            optimization_config['model'] = 'vgg19'
            print(f"content = {content}, style = {style}, model = {optimization_config['model']}")
            neural_style_transfer(optimization_config, NUM_ITERATIONS)
            # ResNet
            set_weights(optimization_config, (100.0, 1000.0, 10.0))
            optimization_config['model'] = 'resnet'
            print(f"content = {content}, style = {style}, model = {optimization_config['model']}")
            neural_style_transfer(optimization_config, NUM_ITERATIONS)

