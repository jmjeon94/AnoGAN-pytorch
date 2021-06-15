import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MNISTLoader(Dataset):
    def __init__(self, mnist, selector):
        self.data_list = []
        self.label_list = []
        for data, label in mnist:
            if label in selector:
                self.data_list.append(data)
                self.label_list.append(label)

    def __getitem__(self, i):
        return self.data_list[i], self.label_list[i]

    def __len__(self):
        return len(self.label_list)

    def check(self):
        for img in self.data_list:
            plt.imshow(img[0], 'gray')
            plt.show()
            if input('exit(x):')=='x':
                break

if __name__=='__main__':

    train_mnist = MNIST(root='./', train=True, transform=T.ToTensor(), download=False)
    test_mnist = MNIST(root='./', train=False, transform=T.ToTensor(), download=False)

    train_loader = DataLoader(dataset=MNISTLoader(train_mnist, [1]),
                              batch_size=256,
                              shuffle=False,
                              drop_last=True)

    test_loader = DataLoader(dataset=MNISTLoader(test_mnist, [4]),
                              batch_size=256,
                              shuffle=False,
                              drop_last=True)

    # check loader showing image 4
    MNISTLoader(test_mnist, [4]).check()