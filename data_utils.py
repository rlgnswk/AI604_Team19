import torch
import torch.nn as nn #
from torch.utils import data
import torchvision
import torch.nn.functional as F

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

def RGB_np2Tensor(img_lr):

    if channel == 1:
        # rgb --> Y (gray)
        img_lr = np.sum(img_lr * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
    # to Tensor
    ts = (2, 0, 1)
    img_lr = torch.Tensor(img_lr.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img_lr = (img_lr / 255.0 - 0.5) * 2
    return img_lr

class train_img:
    def __init__(self, args):
        self.args = args
        self.dirpath = args.datasetPath
        self.filelist = os.listdir(dirpath)
        self.ppath =os.path.join(dirpath, filelist[0])
        self.dirTest='Test'
        self.saveTar=os.path.join(self.dirpath, self.dirTest)

    def make_dataset(self):
        name_bef = os.path.join(ppath, 'im1.png')
        img_bef = cv2.imread(name_bef)
        tm = (1, 2, 0)
        imgTar = RGB_np2Tensor(img_bef)
        imgTar = torch.unsqueeze(imgTar, 0)
        for i in range(100):
            img_aug = augment(imgTar)
            img_aug = torch.squeeze(img_aug , 0)
            img_aug  = img_aug .detach().cpu().numpy()
            img_aug = img_aug .transpose(tm).astype(float)
            cv2.imwrite(saveTar, img_aug )
        return self.saveTar

def downscale(img_aug):
    img_aug=F.interpolate(input=img_aug, scale_factor=0.5, mode='bicubic')

class Dataset(Dataset):
    def __init__(self):
        self.test_dir=train_img
        self.test_file=os.listdir(test_dir)
        self.Testdir=[]
        for i in range(0, len(self.filelist)):
            self.testpath = os.path.join(test_dir, self.test_file[i])
            self.Testdir.append(self.testpath
        self.nTest = len(self.Testdir)

    def __getitem__(self, index):
        args = self.args
        name_bef = self.getFileName(idx)
        img_aug = cv2.imread(name_bef)
        img_lr=downscale(img_aug)

        return RGB_np2Tensor(img_lr, img_aug)

    def __len__(self):
        return self.nTest

    def getFileName(self, idx):
        name_bef = os.path.join(self.dirTrain[idx],'im1.png')
        return name_bef

