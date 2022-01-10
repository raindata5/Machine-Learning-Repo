import os
import tarfile
import urllib.request


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/tree/master/data/"
fashion_path = os.path.join("datasets", "fashion")
list_url =
fashion_urls = [DOWNLOAD_ROOT +]








# ROOT_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# PATH_CIFAR = os.path.join("datasets","cifar-10")

# def fetch_cifar_10_data(root_download = ROOT_DOWNLOAD, path_cifar = PATH_CIFAR):
#     if not os.path.isdir(path_cifar):
#         os.makedirs(path_cifar)
#     path_tgz = os.path.join(path_cifar,"cifar-10-python.tar.gz")
#     urllib.request.urlretrieve(root_download, path_tgz)
#     cifar_tar = tarfile.open(path_tgz)
#     cifar_tar.extractall(path=path_cifar)
#     cifar_tar.close()

# import pandas as pd
# import pandas as pd
# def load_cifar_10_data(path_cifar=PATH_CIFAR):
#     dl_path = os.path.join(path_cifar, "cifar-10-python.csv")
#     return pd.read_csv(dl_path)




# PATH_CIFAR_BATCHES = os.path.join(PATH_CIFAR,"cifar-10-batches-py/")
# file = None
# def unpickle(path_cifar_batches = PATH_CIFAR_BATCHES, the_file = file):
#     import pickle
#     dire = os.path.join(path_cifar_batches, the_file)
#     data_dict = open(dire, "rb")
#     data = data_dict.read()
#     data_dict.close() # the problem is that the data consists of bytes

#     return data






# def load_cifar_10_data(path_cifar_batches=PATH_CIFAR_BATCHES):
#     files = os.listdir(path_cifar_batches)
#     data_list = []

#     for file in files:
#         data_list.append(unpickle(path_cifar_batches = PATH_CIFAR_BATCHES,the_file =file))
#     return files, data_list
