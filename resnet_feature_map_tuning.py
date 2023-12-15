from matplotlib import pyplot as plt
from tqdm import tqdm

import utils.utils as utils
from models.definitions.vgg_nets import ResNet50
from neural_style_transfer import build_loss
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse

def prepare_resnet_model(device, content_feature_map_index):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    experimental = False
    model = ResNet50(requires_grad=False, show_progress=False, content_feature_map_index=content_feature_map_index)

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

def neural_style_transfer_tuning(config, resnet_content_index = 4, num_iterations = 400):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    # initilize with content
    init_img = content_img

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_resnet_model(device, resnet_content_index)

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    content_losses = []
    style_losses = []
    tv_losses = []
    # line_search_fn does not seem to have significant impact on result
    optimizer = LBFGS((optimizing_img,), max_iter=num_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            # print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_iterations, should_display=False)
            content_losses.append(content_loss.detach().cpu().numpy())
            style_losses.append(style_loss.detach().cpu().numpy())
            tv_losses.append(tv_loss.detach().cpu().numpy())

        cnt += 1
        return total_loss

    optimizer.step(closure)

    final_total_loss = config['content_weight'] * content_losses[-1] + config['style_weight'] * style_losses[-1] + config['tv_weight'] * tv_losses[-1]
    print(
        f'Final losses: total loss={final_total_loss:12.4f}, content_loss={config["content_weight"] * content_losses[-1]:12.4f}, style loss={config["style_weight"] * style_losses[-1]:12.4f}, tv loss={config["tv_weight"] *tv_losses[-1]:12.4f}')

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

    # save the plot
    plot_file_name = f'tuning_{utils.generate_out_img_name(config)}'
    plt.savefig(os.path.join(config['plots_dir'], plot_file_name))
    plt.close()

    return content_losses[-1], style_losses[-1], tv_losses[-1]

def feature_map_index_tuning():
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
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19', 'resnet'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
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

    # set hyperparameters
    feature_indices = [_ for _ in range(5)]
    # Tune!
    for feature_index in feature_indices:
        print("Feature index: ", feature_index)
        optimization_config['disambiguater'] = f'tuning_resnet_fmi={feature_index}'
        results_path = neural_style_transfer_tuning(optimization_config, feature_index)

def weight_tuning():
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
    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='candy.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19', 'resnet'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
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

    # set hyperparameters
    content_weights = [10. ** i for i in range(1, 5)]
    style_weights = [10. ** i for i in range(2, 6)]
    tv_weights = [10. ** i for i in range(-1, 2)]

    content_loss_lst = []
    style_loss_lst = []
    tv_loss_lst = []

    hyperparameter_sets = []

    # Tune!
    for content_weight in tqdm(content_weights):
        optimization_config['content_weights'] = content_weight
        for style_weight in style_weights:
            optimization_config['style_weights'] = style_weight
            for tv_weight in tv_weights:
                optimization_config['tv_weights'] = tv_weights
                print(f'content_weight: {content_weight}, style_weight: {style_weight}, tv_weight: {tv_weight}')
                optimization_config['disambiguater'] = f'cw={content_weight}-sw={style_weight}-tw={tv_weight}'
                content_loss, style_loss, tv_loss = neural_style_transfer_tuning(optimization_config)
                content_loss_lst.append(content_loss)
                style_loss_lst.append(style_loss)
                tv_loss_lst.append(tv_loss)
                hyperparameter_sets.append((content_weight, style_weight, tv_weight))

    print(f"(cw, sw, tv) {hyperparameter_sets[np.argmin(content_loss_lst)]} had lowest content loss")
    print(f"(cw, sw, tv) {hyperparameter_sets[np.argmin(style_loss_lst)]} had lowest style loss")
    print(f"(cw, sw, tv) {hyperparameter_sets[np.argmin(tv_loss_lst)]} had lowest tv loss")

    """
    (cw, sw, tv) (10.0, 100.0, 10.0) had lowest content loss
    (cw, sw, tv) (100.0, 1000.0, 10.0) had lowest style loss
    (cw, sw, tv) (100.0, 100.0, 10.0) had lowest tv loss
    """

if __name__ == "__main__":
    weight_tuning()





