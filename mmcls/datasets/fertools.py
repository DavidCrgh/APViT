import os

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples

#                  0       1         2          3       4               5           6           7
FER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']

def convert2coarse_label(i:int):
    """The first four are negative"""
    if i <= 3:
        return 0
    return i - 3


def gen_class_map(dataset_class):
    """
    generate the convert map from DATASET_CLASSES to FER_CLASSES
    """
    convert_map = []
    for i in dataset_class:
        convert_map.append(FER_CLASSES.index(i))
    assert sum(convert_map) == sum([i for i in range(len(dataset_class))])
    return convert_map