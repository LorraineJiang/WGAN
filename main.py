import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable
import os
import sys
import json
import time
from tensorboardX import SummaryWriter

import models.dcgan as dcgan
import models.mlp as mlp
import function.entropy as Entropy
import function.print_save as PrintSave
import function.metrics as metrics

if __name__=="__main__":

    '''----------------------------------------环境选择----------------------------------------'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gantype', required=True, help='wgan| wgangini | wgantsallis | wgangp  | wgangp_gini | wgangp_tsallis ')
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()

    # 选择训练结果的存储位置，默认位置为samples文件夹下
    if opt.experiment is None:
        opt.experiment = 'samples'
    os.makedirs(opt.experiment, exist_ok=True)

    # 打开存储文件，保证在控制台输出的同时也保存到文件
    save_file = opt.experiment + '/output_log.log'
    # os.makedirs(save_file)
    sys.stdout = PrintSave.Logger(filename=save_file, stream=sys.stdout)
    print(opt)

    # 定义tensorboardX实例，以及存储位置:SummaryWriter(存放路径)
    # 添加标量add_scalar(图的名称，Y轴数据，X轴数据)
    # 添加多个标量add_scalars(图的名称，Y轴数据{'名称1'：值，'名称2'：值}，X轴数据)
    writer_scalar = SummaryWriter(log_dir = opt.experiment + '/scalar')
    writer_graph = SummaryWriter(log_dir = opt.experiment + '/graph')

    # 为CPU/GPU设置种子用于生成随机数，以使得结果是确定的
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    # 可能是因为每次迭代都会引入点临时变量，会导致训练速度越来越慢，基本呈线性增长
    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 根据选择的数据集是ImageNet/LSUN/Cifar-10而分类
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        # dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
        dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'mnist':
        # dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
        dataset = dset.MNIST(root=opt.dataroot, download=False,   #是否下载
                             transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, train=True , download=False,   #是否下载
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    assert dataset  # assert用处:让程序测试条件，如果条件正确继续执行，如果条件错误，报错
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers),
                                            drop_last=True)

    ngpu = int(opt.ngpu)    # GPU的数量,默认为1
    nz = int(opt.nz)        # z向量的大小,默认为100
    ngf = int(opt.ngf)      # 生成器大小,默认为64
    ndf = int(opt.ndf)      # 判别器大小,默认为64
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # 写出生成器配置，与训练checkpoints（.pth）一起生成图像
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ndf": ndf, "ngpu": ngpu,
                        "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G, "mlp_D": opt.mlp_D}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    '''----------------------------------------初始化生成器与判别器----------------------------------------'''
    # 定义权重初始化函数：在G网络和D网络上调用自定义权重初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # 在环境设置中是否对DCGAN选择了不要批归一化，默认为True
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:     # 对生成网络G使用MLP，默认为True
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:               # 若不是用MLP，还需要输入G和D上额外的层数，默认为0
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # 写出生成器配置，与训练checkpoints（.pth）一起生成图像
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ndf": ndf, "ngpu": ngpu,
                        "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G, "mlp_D": opt.mlp_D}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)        # 生成器G调用权重初始化函数
    if opt.netG != '':              # 如果生成器G需要加载checkpoint
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mlp_D:                   # 若判别器D选择的是MLP，就调用自写的MLP函数
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:                           # 若判别器D选择的是DCGAN，就调用自写的DCGAN函数，并给判别器D初始化权重
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':              # 如果判别器D需要加载checkpoint
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # 进行类型转换，转为tensor类型
    input = torch.FloatTensor(opt.batchSize, opt.nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    # 若选择了cuda环境
    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # 优化器设置：Adam/RMSProp
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    '''----------------------------------------开始训练----------------------------------------'''
    time_start = time.time()                    # 开始计时
    gen_iterations = 0
    for epoch in range(opt.niter):              # niter为环境设置中，要训练的论述epoch
        data_iter = iter(dataloader)            # iter迭代每一个加载的数据
        i = 0
        while i < len(dataloader):
            ############################
            # (1) 更新判别器网络netD
            ###########################
            for p in netD.parameters(): # 重置requires_grad
                p.requires_grad = True # 在下面的 netG 更新中它们被设置为 False 

            # 在第一个epoch中，将判别器迭代100次，将生成器迭代１次
            # 当生成器迭代次数>=25时，生成器每迭代一次，判别器迭代默认的５次
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                if opt.gantype in ['wgan', 'wgan_gini', 'wgan_tsallis']:
                    # 梯度裁剪，clamp类似于clip：在原对象基础上进行修改，规范到一个区间
                    for p in netD.parameters():
                        p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # 在判别器D中训练real数据
                real_cpu, _ = data
                # 将判别器的优化器参数清零一下，最后再用step()将backward()计算出来的梯度更新
                netD.zero_grad()
                batch_size = real_cpu.size(0)

                real_cpu = real_cpu.cuda() if opt.cuda else real_cpu
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                real = inputv

                netD_predreal, errD_real = netD(inputv)
                errD_real.backward(one)     # backward函数：计算反向传播计算梯度值

                # 在判别器D中训练fake数据
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                with torch.no_grad():       # 完全冻结生成器 netG 
                    noisev = Variable(noise)       
                    fake = Variable(netG(noisev).data)
                inputv = fake
                netD_predfake, errD_fake = netD(inputv)
                errD_fake.backward(mone)    # backward函数,mone是因为最大化目标函数=最大化第一项and最小化第二项

                '''----------------------------------------WGAN_GP的梯度惩罚项----------------------------------------'''
                if opt.gantype in ['wgangp', 'wgangp_gini', 'wgangp_tsallis']:
                    lambda_gp = 10
                    # epsilon是位于0-1之间均匀分布的随机权重项
                    epsilon = torch.rand(real.size(0), 1, 1, 1)        # size(0)为矩阵行数
                    epsilon = epsilon.cuda() if opt.cuda else epsilon
                    # 获取夹在real和fake之间的随机分布
                    x_hat = epsilon * real + (1 - epsilon) * fake
                    x_hat = x_hat.cuda() if opt.cuda else x_hat
                    x_hat = Variable(x_hat, requires_grad=True)
                    _, y_hat = netD(x_hat)
                    # 计算y_hat相对于x_hat的梯度之和
                    grad_outputs = torch.ones(y_hat.size())
                    grad_outputs = grad_outputs.cuda() if opt.cuda else grad_outputs
                    gradients = autograd.grad(
                        outputs=y_hat,
                        inputs=x_hat,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = lambda_gp * torch.mean(((gradients.norm(2, dim=1) - 1) ** 2))
                    gradient_penalty.backward(mone)
                
                '''----------------------------------------关于熵的相关改动----------------------------------------'''
                # 根据训练gan的类型选择是否加入Gini熵
                if opt.gantype in ['wgangini', 'wgangp_gini']:
                    lambda_gini = 0.1
                    gini_fake = Entropy.Gini_Impurity(netD_predfake, lambda_gini).value()

                # 根据训练gan的类型选择是否加入Tsallis熵(q必须为≠1的非负实数)
                if opt.gantype in ['wgantsallis', 'wgangp_tsallis']:
                    q = 2
                    lambda_tsallis = 0.01
                    tsallis_fake = Entropy.Tsallis_Entropy(netD_predfake, lambda_tsallis, 2).value()

                # 加入Tsallis相对熵(类似于KL散度，要求q必须为≠1的非负实数)
                # lambda_tsallis_relative = 0.01
                # tsallis_relative_fake = entropy.Tsallis_Relative_Entropy(预测标签, 真实标签, lambda_tsallis_relative).value()

                '''----------------------------------------计算判别器总误差‘梯度’并更新----------------------------------------'''
                errD = Variable
                if opt.gantype == 'wgan':errD = errD_real - errD_fake
                elif opt.gantype == 'wgangini':errD = errD_real - errD_fake + gini_fake
                elif opt.gantype == 'wgantsallis':errD = errD_real - errD_fake + tsallis_fake
                elif opt.gantype == 'wgangp':errD = errD_real - errD_fake + gradient_penalty
                elif opt.gantype == 'wgangp_gini':errD = errD_real - errD_fake + gradient_penalty + gini_fake
                elif opt.gantype == 'wgangp_tsallis':errD = errD_real - errD_fake + gradient_penalty + tsallis_fake
                optimizerD.step()           # 进行梯度更新

            ############################
            # (2) 更新生成器网络netG
            ###########################
            for p in netD.parameters():
                p.requires_grad = False             # 避免计算 
            # 将生成器的优化器参数清零一下，最后再用step()将backward()计算出来的梯度更新
            netG.zero_grad()
            # 如果当前是最后一批数据，确保一个batch的完整
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            _, errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            ############################
            # (3) 输出相关的训练数据
            ###########################
            if i % 50 == 0 :
                # 计算wasserstein距离，输出并可视化
                wasserstein_distance = metrics.wasserstein_distance(errD_real, errD_fake)
                # 可视化Loss_D,Loss_G,Loss_D_real,Loss_D_fake,及数据输出和存储
                writer_scalar.add_scalar('Loss_D', errD.data[0], i)
                writer_scalar.add_scalar('Loss_G', errG.data[0], i)
                writer_scalar.add_scalar('Wasserstein Distance', wasserstein_distance, i)
                writer_scalar.add_scalars('Loss_D_real_fake', {'Loss_D_real' : errD_real.data[0], 'Loss_D_fake' : errD_fake.data[0]}, i)
                print_str = '[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake: %f'\
                    % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0])
                # 根据训练gan的类型具体可视化和输出
                if opt.gantype == 'wgangini':
                    writer_scalar.add_scalar('Gini_fake', gini_fake.item(), i)
                    print_str = print_str + ' Gini_fake: %f' % (gini_fake.item())
                elif opt.gantype == 'wgantsallis':
                    writer_scalar.add_scalar('Tsallis_fake', tsallis_fake.item(), i)
                    print_str = print_str + ' Tsallis_fake: %f' % (tsallis_fake.item())
                elif opt.gantype == 'wgangp':
                    writer_scalar.add_scalar('Gradient_Penalty', gradient_penalty.item(), i)
                    print_str = print_str + ' Gradient_Penalty: %f' % (gradient_penalty.item())
                elif opt.gantype == 'wgangp_gini':
                    writer_scalar.add_scalar('Gradient_Penalty', gradient_penalty.item(), i)
                    writer_scalar.add_scalar('Gini_fake', gini_fake.item(), i)
                    print_str = print_str + ' Gradient_Penalty: %f' % (gradient_penalty.item())
                    print_str = print_str + ' Gini_fake: %f' % (gini_fake.item())
                elif opt.gantype == 'wgangp_tsallis':
                    writer_scalar.add_scalar('Gradient_Penalty', gradient_penalty.item(), i)
                    writer_scalar.add_scalar('Tsallis_fake', tsallis_fake.item(), i)
                    print_str = print_str + ' Gradient_Penalty: %f' % (gradient_penalty.item())
                    print_str = print_str + ' Tsallis_fake: %f' % (tsallis_fake.item())
                print(print_str)

            # 每500iterations就输出并保存一次生成效果
            if gen_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
                with torch.no_grad():       # 完全冻结生成器 netG 
                    fake = netG(Variable(fixed_noise))
                if opt.nc == 3:
                    inception_score_mean, inception_score_std = metrics.inception_score(imgs=fake, cuda=True, batch_size=64, resize=True, splits=1)
                    print('[Generator iterations: %d] Inception_Score: %f ± %f' % (gen_iterations, inception_score_mean, inception_score_std))
                    writer_scalar.add_scalar('Inception_score', inception_score_mean, gen_iterations)
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        # 保存模型(仅保存模型参数，若保存整个模型则去掉.state_dict())
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

    time_end = time.time()
    print('迭代结束，耗时：%.2f秒'%(time_end-time_start))
    writer_scalar.close()
    writer_graph.close()