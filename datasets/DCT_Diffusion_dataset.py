# from datasets.transcg import TransCG
# from datasets.cleargrasp import ClearGraspRealWorld, ClearGraspSynthetic
# from datasets.omniverse_object import OmniverseObject
# from datasets.transparent_object import TransparentObject
# from datasets.mixed_dataset import MixedDataset

from .transcg import TransCG
from .cleargrasp import ClearGraspRealWorld, ClearGraspSynthetic
from .omniverse_object import OmniverseObject
from .transparent_object import TransparentObject
from .mixed_dataset import MixedDataset

from torch.utils.data import ConcatDataset
import logging
import yaml
from .logger import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


def get_dataset(dataset_params = None, split = 'train'):

    if dataset_params is None:
        dataset_params = dataset_params
    dataset_params = dataset_params.get(split, {'type': 'transcg'})
    if type(dataset_params) == dict:
        dataset_type = str.lower(dataset_params.get('type', 'transcg'))
        if dataset_type == 'transcg':
            dataset = TransCG(split = split, **dataset_params)
        elif dataset_type == 'cleargrasp-real':
            dataset = ClearGraspRealWorld(split = split, **dataset_params)
        elif dataset_type == 'cleargrasp-syn':
            dataset = ClearGraspSynthetic(split = split, **dataset_params)
        elif dataset_type == 'omniverse':
            dataset = OmniverseObject(split = split, **dataset_params)
        elif dataset_type == 'transparent-object':
            dataset = TransparentObject(split = split, **dataset_params)
        elif dataset_type == 'mixed-object':
            dataset = MixedDataset(split = split, **dataset_params)
        else:
            raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
        logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
    elif type(dataset_params) == list:
        dataset_types = []
        dataset_list = []
        for single_dataset_params in dataset_params:
            dataset_type = str.lower(single_dataset_params.get('type', 'transcg'))
            if dataset_type in dataset_types:
                raise AttributeError('Duplicate dataset found.')
            else:
                dataset_types.append(dataset_type)
            if dataset_type == 'transcg':
                dataset = TransCG(split = split, **single_dataset_params)
            elif dataset_type == 'cleargrasp-real':
                dataset = ClearGraspRealWorld(split = split, **single_dataset_params)
            elif dataset_type == 'cleargrasp-syn':
                dataset = ClearGraspSynthetic(split = split, **single_dataset_params)
            elif dataset_type == 'omniverse':
                dataset = OmniverseObject(split = split, **single_dataset_params)
            elif dataset_type == 'transparent-object':
                dataset = TransparentObject(split = split, **single_dataset_params)
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
            dataset_list.append(dataset)
            logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
        dataset = ConcatDataset(dataset_list)
    else:
        raise AttributeError('Invalid dataset format.')
    return dataset


# if __name__ =='__main__':
#     cfg_filename = '/home/zl/zl_dev/Diffusion/DiffusionDepth/src/configs/train_cgsyn+ood_val_cgreal.yaml'
#     with open(cfg_filename, 'r') as cfg_file:
#         cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
#     dataset_params = cfg_params.get('dataset', {'data_dir': 'data'})
#     train_dataset = get_dataset(dataset_params,split='train')
#     test_dataset = get_dataset(dataset_params,split='test')
#     # get_dataset(dataset_params)
#     pass
        # builder = ConfigBuilder(**cfg_params)