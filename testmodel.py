import json
import torch
import models.dcgan as dcgan
import models.mlp as mlp
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

'''-----------------------从generator_config.json文件里读取判别器和生成器的类型-----------------------'''
dataroot = './dataset/mnist'
experiment = './samples/wgantest/'
config_file = experiment + 'generator_config.json'
with open(config_file, 'r') as gencfg:
    generator_config = json.loads(gencfg.read())

imageSize = generator_config["imageSize"]
nz = generator_config["nz"]
nc = generator_config["nc"]
ngf = generator_config["ngf"]
ndf = generator_config["ndf"]
noBN = generator_config["noBN"]
ngpu = generator_config["ngpu"]
mlp_G = generator_config["mlp_G"]
mlp_D = generator_config["mlp_D"]
n_extra_layers = generator_config["n_extra_layers"]

if noBN:
    netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif mlp_G:
    netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
if mlp_D:                   # 若判别器D选择的是MLP，就调用自写的MLP函数
    netD = mlp.MLP_D(imageSize, nz, nc, ndf, ngpu)
else:                           # 若判别器D选择的是DCGAN，就调用自写的DCGAN函数，并给判别器D初始化权重
    netD = dcgan.DCGAN_D(imageSize, nz, nc, ndf, ngpu, n_extra_layers)

'''-----------------------读取训练好的模型参数并生成模型-----------------------'''
# pretrained=True就可以使用预训练的模型
pthfile_D = experiment + 'netD_epoch_0.pth'
pthfile_G = experiment + 'netG_epoch_0.pth'
netD.load_state_dict(torch.load(pthfile_D))
netG.load_state_dict(torch.load(pthfile_G))

'''-----------------------可视化模型-----------------------'''
# 测试生成器网络
batchSize, nz = 64, 100
noise = torch.FloatTensor(batchSize, nz, 1, 1)
noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
# 定义tensorboardX实例，以及存储位置:SummaryWriter(存放路径)
with SummaryWriter(logdir=experiment+'graphG', comment='NetG') as w:
    w.add_graph(netG, noise)
fake = netG(noise)
fake.data = fake.data.mul(0.5).add(0.5)
vutils.save_image(fake.data, '{0}/fake_testsamples.png'.format(experiment))

# 测试判别器网络    判断效果不能用acc！！！！！！！！有误！！！！！
imageSize = 64
dataset = dset.MNIST(root=dataroot, download=False,   #是否下载
                     train=False,
                     transform=transforms.Compose([
                     transforms.Scale(imageSize),
                     transforms.CenterCrop(imageSize),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5]),
            ]))
test_loader = torch.utils.data.DataLoader(dataset, batchSize, shuffle=True)
for epoch in range(1):
    correct, total = 0, 0
    data_iter = iter(test_loader)
    realimage, label = data_iter.next()
    outputs, _ = netD(realimage)
    print('!!!outputs!!!!!',outputs)
    _, pred = torch.max(outputs, 1)
    print('!!!pred label!!!!!',pred,label)
    correct += (pred == label).sum().float()
    total += len(label)
    print('!!!correct total!!!!!',correct.item(),total)
    acc = correct / total
    print('!!!acc!!!!!',acc.item())
    with SummaryWriter(logdir=experiment+'graphD', comment='NetD') as w:
        w.add_graph(netD, realimage)