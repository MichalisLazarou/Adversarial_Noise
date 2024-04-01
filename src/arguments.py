import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for inference')
    # load pretrained model
    parser.add_argument('--target_class', type=str, default='electric ray', help='Supported classes are all the classes supported by ImageNet, found in src.supported_classes.py')
    parser.add_argument('--img_path', type=str, default='/home/michalislazarou/Downloads/lion.jpg', help='Provide absolute path to image')
    parser.add_argument('--attack', type=str, default='EAD', choices=['PGD', 'EAD'], help='Supported adversarial attacks')
    opt = parser.parse_args()
    return opt

