import torch
from data_loader import DataTest
import argparse
from model.pa.model_mpa_cat import AF
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
def visualize_single_feature_map(feature,name):
    plt.imshow(feature)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar()
    plt.savefig('plt/attention_map/'+name+".png",bbox_inches = 'tight',dpi=500) # 保存图像到本地
    plt.show()
def test(test_dataloader,args):
    device=args.device
    net = AF(args).to(device)
    print(net)
    net.load_state_dict(torch.load(args.saveModel_dir,map_location='cpu'))
    net.eval()
    t1=time.time()
    for i_test, (image1,image2) in enumerate(test_dataloader):
        image1=image1.to(device)
        image2=image2.to(device)
        out,F1_n_a,F2_n_a,F1_a,F2_a,mask1,mask2=net(image1,image2)
        out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))
        out_image.save("result/PAF/pa/50/"+"color_lytro_"+str(i_test+1).zfill(2) + ".png")
    t2=time.time()
    print(t2-t1)

def test_vision(test_dataloader,args):
    device=args.device
    net = AF(args).to(device)
    print(net)
    net.load_state_dict(torch.load(args.saveModel_dir,map_location='cpu'))
    net.eval()
    for i_test, (image1,image2) in enumerate(test_dataloader):
        image1=image1.to(device)
        image2=image2.to(device)
        im1=net.conv1(image1)
        im2=net.conv1(image2)
        im1=net.rasb(im1)
        im2=net.rasb(im2)
        im1=net.conv2(im1)
        im2=net.conv2(im2)
        pa1=net.pa.pa2(im1)
        pa2=net.pa.pa2(im2)
        EPSILON = 1e-10
        mask1 = torch.exp(pa1) / (torch.exp(pa1) + torch.exp(pa2) + EPSILON)
        mask2 = torch.exp(pa2) / (torch.exp(pa1) + torch.exp(pa2) + EPSILON)
        temp_mask_sp_1= torch.squeeze(mask1.data.cpu(), 0)
        temp_mask_sp_2 = torch.squeeze(mask2.data.cpu(), 0)
        if i_test>=4:
            visualize_single_feature_map(temp_mask_sp_1[0, :, :], str(i_test + 1).zfill(2) + "_image1_attention_sp1")
            visualize_single_feature_map(temp_mask_sp_2[0, :, :], str(i_test + 1).zfill(2) + "_image1_attenton_sp2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testData_dir', type=str, default="E:/E2EMutil/DataSet/testData/lytro")
    parser.add_argument('--saveModel_dir', type=str, default='experment_mpa/pa/epoch_50_loss_4.276947.pth')
    parser.add_argument('--result', type=str, default='result')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir,transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)

