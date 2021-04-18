from model.max.model_max import AF
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader import SalObjDataset
from lp_lssim_loss import LpLssimLoss
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.nn import init
import numpy as np
import random
import time

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def train(train_loader,args):
    device=args.device
    n_epochs=args.n_epochs
    model_dir=args.saveModel_dir
    batch_size=args.batch_size
    train_num=15570
    writer1 = SummaryWriter(log_dir="log/loss")
    # ------- 1. define model --------
    # define the net
    net = AF(args).to(device)
    net.apply(weights_init_xavier)
    #define the loss

    criterion = LpLssimLoss().to(args.device)
    L1=torch.nn.L1Loss().to(args.device)
    # MSE=torch.nn.MSELoss().to(args.device)
    # ------- 2. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------- 3. training process --------
    # ite_num = 0
    running_loss = 0.0
    # ite_num4val = 0

    for epoch in range(0, n_epochs):
        net.train()
        t1 = time.time()
        for i, (image1,image2,label) in enumerate(train_loader):

            image1=image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            # ite_num = ite_num + 1
            # ite_num4val = ite_num4val + 1
            output,F1_n_a,F2_n_a,x1,x2,F1,F2=net(image1,image2)
            l1=L1(output,label)
            # mssim=1-ms_ssim( out, label, data_range=1, size_average=True)
            ssim=criterion(output,label)
            loss =0.9*ssim+0.1*l1
            # writer1.add_scalar('ssim', ssim, i)
            # writer1.add_scalar('l1', l1, i)
            # writer1.add_scalar('total', loss, i)
            # if i==0:
            #     im1=image1[0].detach().cpu()
            #     im2=image2[0].detach().cpu()
            #     lab1=label[0].detach().cpu()
            #     out_image=out[0].detach().cpu()
            #     writer1.add_image('resutl', make_grid([im1,im2,lab1,out_image], nrow=5, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F1_ori', make_grid(F1_ori[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F2_ori', make_grid(F2_ori[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F1_att', make_grid(F1_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
            #     writer1.add_image('F2_att', make_grid(F2_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
                # writer1.add_image('fusion', make_grid(fusion[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
                # writer1.add_image('F1_a', make_grid(F1_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
                # writer1.add_image('F3_a', make_grid(F3_a[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)
                # writer1.add_image('atten_fusion', make_grid(atten_fusion[0].detach().cpu().unsqueeze(dim=1), nrow=8, padding=20, normalize=False, pad_value=1), epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f,ssim:%8f,l1:%8f " % (epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item(),ssim.item(),l1.item()))
        # writer1.add_scalar('训练损失', running_loss, epoch)
        torch.save(net.state_dict(), model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss))
        running_loss = 0.0
        scheduler.step()
        t2 = time.time()
        print(t2-t1)

    print('-------------Congratulations! Training Done!!!-------------')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3,help='rgb is 3,gray is 1')
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    parser.add_argument('--dataset_dir', type=str, default="E:/E2EMutil/DataSet/trainData")
    parser.add_argument('--saveModel_dir', type=str, default='experment_mpa/max/')
    return parser.parse_args()


if __name__ == '__main__':
    training_setup_seed(1)
    args = parse_args()
    transforms_ = [transforms.Resize((256,256), Image.BICUBIC),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    train_set = SalObjDataset(dataset_dir=args.dataset_dir,transforms_=transforms_,rgb=True)
    salobj_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train(salobj_dataloader, args)
