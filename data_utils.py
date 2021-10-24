import torch 
import torch.nn as nn #
from torch.utils import data
import torchvision


def dataAug(imglr):
    if random.random() < 0.3:  # horizontal flip
        imglr = imglr[:, ::-1, :]


    if random.random() < 0.3:  # vertical flip
        imglr =  imglr[::-1, :, :]

    rot = random.randint(0, 3)  # rotate
    imglr = np.rot90(imglr, rot, (0, 1))

    crop_size_w=random.randint(0,40)
    crop_size_h=random.randint(0,40)

    input_size_h, input_size_w, _ = imglr[0].shape
    x_start = random.randint(0, input_size_w - crop_size_w)
    y_start = random.randint(0, input_size_h - crop_size_h)

    img_lr = imglr[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]

    return  img_lr




def RGB_np2Tensor(img_lr, img_aug):

    if channel == 1:
        # rgb --> Y (gray)
        img_lr = np.sum(img_lr * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        img_aug = np.sum(img_aug * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    img_lr = torch.Tensor(img_lr.transpose(ts).astype(float)).mul_(1.0)
    img_aug = torch.Tensor(img_aug.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img_lr = (img_lr / 255.0 - 0.5) * 2
    img_aug = (img_aug / 255.0 - 0.5) * 2

    return img_lr, img_aug


def downscale(img_aug):
    






class Dataset(Dataset):
    def __init__(self):
        self.args = args
        dirpath = args.datasetPath
        self.filelist = os.listdir(dirpath)
        self.dirTrain = []
        for i in range(0, len(self.filelist)):
            self.ppath = os.path.join(dirpath, self.filelist[i])
            self.trainlist = os.listdir(self.ppath)
            for j in range(0, len(self.trainlist)):
                self.dirTrain.append(os.path.join(self.ppath, self.trainlist[j]))

        self.nTrain = len(self.dirTrain)

    def __getitem__(self, index):
        args = self.args
        name_bef = self.getFileName(idx)
        img_bef = cv2.imread(name_bef)

        img_aug = augment(img_bef)
        img_lr=downscale(img_aug)

        return RGB_np2Tensor(img_lr, img_aug)

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name_bef = os.path.join(self.dirTrain[idx],'im1.png')
        return name_bef
