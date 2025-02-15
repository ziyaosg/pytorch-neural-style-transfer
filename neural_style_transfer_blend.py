import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config, lbfgs_iterations = 1000):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    # style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    # validate --style_img_name
    style_image_input = [i.strip() for i in config['style_img_name'].split(',')]
    style_image_list = list()
    ext = [".jpg", ".jpeg", ".png", ".tiff"]
    for each_input in style_image_input:
        assert(os.path.splitext(each_input)[1].lower() in ext), \
            '{} is not a valid image file!'.format(each_input)
        style_image_list.append(each_input)

    # validate --style_blend_weights
    if config['style_blend_weights'] is None:
        style_blend_weights = [1] * len(style_image_input)
    else:
        style_blend_weights = [float(w.strip()) for w in config['style_blend_weights'].split(',')]
        assert(len(style_image_input) == len(style_blend_weights)), \
            '--style_blend_weights and --style_img_name must have the same number of elements!'

    # normalize the style blending weights so they sum to 1
    style_blend_sum = sum(style_blend_weights)
    style_blend_weights = [w / style_blend_sum for w in style_blend_weights]

    # compute output path
    style_name_combined = ''
    for name, weight in zip(style_image_list, style_blend_weights):
        style_name_combined += '_' + name.split('.')[0] + '_' + str(weight)
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + style_name_combined
    out_dir_name = out_dir_name[:50]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # prepare content and style images
    content_img = utils.prepare_img(content_img_path, config['height'], device)
    # style_img = utils.prepare_img(style_img_path, config['height'], device)
    style_imgs = list()
    for img in style_image_list:
        img_path = os.path.join(config['style_images_dir'], img)
        style_imgs.append(utils.prepare_img(img_path, config['height'], device))


    # initialization method
    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        # style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        style_index = config['init_style_index'] - 1
        style_path = os.path.join(config['style_images_dir'], style_image_list[style_index])
        style_img_resized = utils.prepare_img(style_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized
    
    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)

    # ---------- below is the essential component to handle style blends ----------
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = [neural_net(img) for img in style_imgs]

    
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    
    num_layers = len(style_feature_maps_indices_names[0])
    target_style_representations = []

    for style_features in style_img_set_of_feature_maps:
        # compute and store the Gram matrix for each specified layer of this style image
        for layer_index in style_feature_maps_indices_names[0]:
            layer_feature_maps = style_features[layer_index]
            gram_matrix = utils.gram_matrix(layer_feature_maps)
            target_style_representations.append(gram_matrix)
    
    layer_wise_style_representations = [[] for _ in range(num_layers)]

    # group Gram matrices by layer
    for i, gram_matrix in enumerate(target_style_representations):
        layer_index = i % num_layers
        layer_wise_style_representations[layer_index].append(gram_matrix)

    # blend Gram matrices layer-wise
    blended_style_representations = []
    for layer_reps in layer_wise_style_representations:
        blended_rep = sum([weight * rep for weight, rep in zip(style_blend_weights, layer_reps)])
        blended_style_representations.append(blended_rep)

    target_representations = [target_content_representation, blended_style_representations]
    # ---------- above is the essential component to handle style blends ----------

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": lbfgs_iterations,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    content_losses = []
    style_losses = []
    tv_losses = []
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
                content_losses.append(content_loss.detach().cpu().numpy())
                style_losses.append(style_loss.detach().cpu().numpy())
                tv_losses.append(tv_loss.detach().cpu().numpy())

            cnt += 1
            return total_loss

        optimizer.step(closure)

        # graph losses
        content_losses_np = np.array(content_losses)
        style_losses_np = np.array(style_losses)
        tv_losses_np = np.array(tv_losses)

        # Create an array for the x-axis (iterations)
        iterations = np.arange(len(content_losses_np))

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, content_losses_np, label='Content Loss')
        plt.plot(iterations, style_losses_np, label='Style Loss')
        plt.plot(iterations, tv_losses_np, label='Total Variation Loss')

        # Set the y-axis to log scale
        plt.yscale('log')

        # Add labels and title
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses for Style Transfer Neural Network')
        plt.legend()

        # Show the plot
        plot_file_name = f'training_{utils.generate_out_img_name(config)}'
        plt.savefig(os.path.join(config['plots_dir'], plot_file_name))
        plt.close()

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
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
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e1)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19', 'resnet'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--init_style_index", type=int, help="choosing which style image as initialization (index starts with 1)", default=1)
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
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
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
