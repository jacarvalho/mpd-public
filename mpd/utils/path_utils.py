import os
import os.path as osp

data_directory = os.path.abspath(os.path.dirname(__file__ + '/../../../../'))


# def makedirs(dirname):
#    if not os.path.exists(dirname):
#        os.makedirs(dirname)

def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_pebm_src():
    # directory = os.environ['PEBM_DATA_DIR']
    makedirs(data_directory)
    return data_directory


def get_pebm_data_dir():
    directory = osp.join(get_pebm_src(), 'data')
    makedirs(directory)
    return directory


def get_pebm_mesh_density_dir():
    directory = osp.join(get_pebm_data_dir(), 'input_objs')
    makedirs(directory)
    return directory


def get_pebm_pointcloud_occupancy_dir():
    directory = osp.join(get_pebm_data_dir(), 'train_data')
    makedirs(directory)
    return directory


if __name__ == '__main__':
    print(get_pebm_src())
    print(get_pebm_data_dir())
