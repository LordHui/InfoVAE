# all__ = ['dataset_celeba', 'dataset_cifar', 'dataset_imagenet', 'dataset_mnist', 'dataset_svhn', 'dataset_mog', 'dataset_lsun', 'dataset_coco']
from .dataset import *
from .dataset_celeba import CelebADataset
from .dataset_cifar import CifarDataset
from .dataset_imagenet import ImagenetDataset
from .dataset_mnist import MnistDataset
from .dataset_svhn import SVHNDataset
from .dataset_mog import MoGDataset
from .dataset_lsun import LSUNDataset
from .dataset_coco import CocoDataset