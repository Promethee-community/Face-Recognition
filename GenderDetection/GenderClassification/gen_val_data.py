import gen_data
import os
# we are only going to use 4 attributes
#sklearn.neural_network.multilayer_perceptron()
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
Flagsnap=True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    #train_dir=[("../Dataset/test","../Dataset/list_attr_celeba.txt"),
    #           ("../Dataset/test2","../Dataset/coco_open.txt")]
    train_dir=[("../Dataset/test2","../Dataset/coco_open.txt")]
    gen_data.gen_data(train_dir,"X_val.npy","Y_val.npy")


if __name__ == "__main__":
    main()