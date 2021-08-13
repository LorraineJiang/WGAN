import json
import torch
import torchvision.models as models
import models.dcgan as dcgan
import models.mlp as mlp
import main as m

'''-----------------------从generator_config.json文件里读取判别器和生成器的类型-----------------------'''
config_file = './samples/generator_config.json'
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
pthfile_D = './samples/netD_epoch_0.pth'
pthfile_G = './samples/netG_epoch_0.pth'
netD.load_state_dict(torch.load(pthfile_D))
netG.load_state_dict(torch.load(pthfile_G))
print(netD)
print(netG)