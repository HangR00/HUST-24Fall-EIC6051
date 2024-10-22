import argparse
import os
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        # print('models',models)
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path

def parse_arguments(description="NN", custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default="cuda", help='Device for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    return args

def get_args():
    '''
    input arguments
    '''
    custom_parameters = [
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": 0,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
    ]
    # parse arguments
    args = parse_arguments(
        description="MNIST", custom_parameters=custom_parameters
    )
    return args

def translate_image(img, translate, direction):
    if(direction == 'left'):
        img_translated = img.transform((img.width, img.height), Image.AFFINE, (1, 0, translate, 0, 1, 0))
    elif(direction == 'right'):
        img_translated = img.transform((img.width, img.height), Image.AFFINE, (1, 0, -translate, 0, 1, 0))
    elif(direction == 'up'):
        img_translated = img.transform((img.width, img.height), Image.AFFINE, (1, 0, 0, 0, 1, translate))
    elif(direction == 'down'):
        img_translated = img.transform((img.width, img.height), Image.AFFINE, (1, 0, 0, 0, 1, -translate))
    return img_translated
    
def rotate_image(img, angle):
    scale_width = int(abs(img.width * np.cos(np.radians(angle))) + abs(img.height * np.sin(np.radians(angle))))
    scale_height = int(abs(img.width * np.sin(np.radians(angle))) + abs(img.height * np.cos(np.radians(angle))))        

    img_rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=0)

    # crop the image
    img_rotated = img_rotated.crop((0, 0, scale_height, scale_width))
    return img_rotated

def scale_image(img, width_scale, height_scale):
    new_height = int(img.height * height_scale)
    new_width = int(img.width * width_scale)
    new_img = Image.new('L', (img.width, img.height), 0)
    img_scaled = img.resize((new_width, new_height))
    # scale the image
    for i in range(int(0+ 0.5* img.width- 0.5*new_width ), int(0+ 0.5* img.width+ 0.5*new_width)):
        for j in range(int(0+ 0.5* img.height- 0.5*new_height), int(0+ 0.5* img.height+ 0.5*new_height)):
            new_img.putpixel((i, j), img_scaled.getpixel((i- int(0+ 0.5* img.width- 0.5*new_width ), j- int(0+ 0.5* img.height- 0.5*new_height))) )
    return new_img