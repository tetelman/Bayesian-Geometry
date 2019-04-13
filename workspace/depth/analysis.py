import sys
import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

trans2tensor = transforms.ToTensor()
trans2resize = transforms.Resize

def image_load_t(path2img, resize=None, log=False):
    img = Image.open(path2img)
    if resize is not None:
        do_resize = transforms.Resize(resize)
        img1 = do_resize(img)
    else:
        img1 = img
    img_t = trans2tensor(img1).unsqueeze(0)
    if log:
        print(' image loaded from', path2img, 'img_size', img_t.size())
    return img_t

def image_save_np(np_img, path2dir, name, log=False):
    path2save = os.path.join(path2dir, name)
    print(np_img.shape)
    img = Image.fromarray((np_img*255.).astype(np.uint8))
    img.save(path2save)
    if log:
        print('image saved to ', path2save, 'img_shape', np_img.shape)

def image_save_t(t_img, path2dir, name, log=False):
    path2save = os.path.join(path2dir, name)
    assert t_img.size(0) == 1
    if t_img.size(1) == 1:
        np_img = (255*t_img[0][0].detach()).numpy().astype(np.uint8)
    else:
        np_img = (255*t_img[0].detach()).numpy().transpose([1, 2, 0]).astype(np.uint8)
# print(np_img.shape)
    img = Image.fromarray(np_img)
    img.save(path2save)
    if log:
        print('image saved to ', path2save)


def get_files(dirname, log=False):
    allfiles = os.listdir(dirname)
    files = []
    for f in allfiles:
        f_path = os.path.join(dirname, f)
        if os.path.isfile(f_path):
            files.append(f_path)

    files.sort()
    if log:
        print('files:', files)
    return files

def load_all_images_np(files, resize=None):
    images = [0]*len(files)
    for i, f in enumerate(files):
        img = Image.open(f)
        if resize is not None:
            img = img.resize(resize)
        img_np = np.array(img).astype(np.float32)/255.
        images[i] = dict(img=img_np, path=f)
# print(img.size)
# print(img_np.shape)
    return images


def load_all_images_t(files, resize=None):
    t_images = [0]*len(files)
    for i, f in enumerate(files):
        img = image_load_t(f, resize=resize, log=True)
        t_images[i] = dict(img=img, path=f)

    return t_images

def make_gauss_kernel(num_ch, ksize, std=None):
    assert ((ksize - 1)//2)*2 == ksize - 1
    if std is None:
        std = ksize/4
    k = torch.ones(1, 1, ksize, ksize)
    ks2 = ksize//2
    var1 = 0.5/(std**2)
    for i in range(ksize):
        for j in range(ksize):
            k[0][0][i][j] = np.exp(-var1*((i-ks2)**2 + (j-ks2)**2))
    k *= (1./k.sum())
    k1 = torch.cat([k]*num_ch)
    return k1

def make_grad_kernel(num_ch, ksize=3):
    assert ksize ==  3
    g = torch.FloatTensor([-1, 0, 1])
    a = torch.FloatTensor([1, 2, 1])
    gx = a.view(3, 1)*g.view(1, 3)
    gy = g.view(3, 1)*a.view(1, 3)
# print(gx.size(), '\n', gx, gy.size(), '\n', gy)
    kernel = torch.cat([gy.view(1, 1, 3, 3), gx.view(1, 1, 3, 3)], 0)
    kernel = torch.cat([kernel]*num_ch, 0)
# print(kernel, kernel.size())
    return kernel

def get_grad_map(img, kernel_=None):
    num_ch = img.size(1)
    if kernel_ is None:
        kernel = make_grad_kernel(num_ch)
    else:
        kernel = kernel_
    groups = num_ch
    padding = (kernel.size(2) - 1)//2
    img1 = F.conv2d(img, kernel, padding=padding, groups=groups)
# print(img1.size())
    return img1

def get_blur(img, kernel):
    ksize = kernel.size(2)
    padding = (ksize-1)//2
    num_ch = img.size(1)
    img = F.conv2d(img, kernel, padding=padding, groups=num_ch)
    return img

def make_grid(size, log=False):
    affine = torch.eye(2, 3).unsqueeze(0)
    grid_size = (1, 1, size[0], size[1])
    grid = F.affine_grid(affine, grid_size)
    print('grid.size=', grid.size())
    return grid

def find_depth_rgb(t_images, dir_out):
    size = list(t_images[0]['img'].size()[2:])
# define parameters
    a2 = nn.Parameter(torch.zeros(1, 2, 2, requires_grad=True))
    a1 = nn.Parameter(torch.zeros(1, 2, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(2, 1, requires_grad=True))
    c1 = nn.Parameter(torch.ones(1, 2, requires_grad=True))
    d0 = nn.Parameter(torch.ones(1, 1, requires_grad=True))
    d1 = nn.Parameter(torch.zeros(1, 1, size[0], size[1], requires_grad=True))

    params = nn.ParameterList([a2, a1, b1, c1, d0, d1])

    grid = make_grid(size)

    d1_grid = F.interpolate(d1, size, mode='bilinear').squeeze(1).unsqueeze(3)
    print(d1_grid.size())

    flow = (grid + grid.matmul(a2) + a1 + d1_grid.matmul(c1)) / (1 + grid.matmul(b1) + d1_grid.matmul(d0))

    print('flow.size=', flow.size())

    warped = F.grid_sample(t_images[0]['img'], flow)
    print('warped.size=', warped.size())

    optimizer = optim.SGD(params, lr = 0.001)

    num_epochs = 10
    num_steps = 1000
    print(' num_epochs=', num_epochs, 'num_steps=', num_steps)
    aloss = 1

    num_ch = t_images[0]['img'].size(1)

    ksize0 = 5
    for epoch in range(1, num_epochs+1):
        ksize = 1 + 2*int((ksize0 - 1)/2 * 1./(1 + 0*(epoch-1)))
        std = ksize//4
        kernel = make_gauss_kernel(num_ch, ksize, std)
        img = t_images[1]['img']
        img = get_blur(img, kernel)

        for s in range(num_steps):

            d1_grid = F.interpolate(d1, size, mode='bilinear').squeeze(1).unsqueeze(3)
            flow = (grid + grid.matmul(a2) + a1 + d1_grid.matmul(c1)) / (1 + grid.matmul(b1) + d1_grid.matmul(d0))
            warped = F.grid_sample(t_images[0]['img'], flow)
            warped = get-blur(warped, kernel)
            loss = (img - warped).abs().mean()
            eloss = loss.item()
            aloss += (eloss - aloss)*0.1
            optimizer.zero_grad()
            loss /= (aloss + 1e-8)
            loss.backward()
            optimizer.step()

        if epoch % (num_epochs/10) == 0 or epoch == 1 or epoch == num_epochs:
            print('epoch=', epoch, 'loss=', aloss,)
# 'd1 min,max:', d1.detach().min().item(),
# d1.detach().max().item(), d1.grad.size(), d1.grad.abs().max().item())
            '''for p in params:
                print(p.grad.abs().max().item())'''
        image_save_t((warped.detach() - img).abs(), dir_out, 'diff_warped.png', log=True)

    d1_grid = F.interpolate(d1, size, mode='bilinear').squeeze(1).unsqueeze(3)
    flow = (grid + grid.matmul(a2) + a1 + d1_grid.matmul(c1)) / (1 + grid.matmul(b1) + d1_grid.matmul(d0))
    warped = F.grid_sample(t_images[0]['img'], flow)
    diff_warped_one = (warped.detach() - t_images[1]['img']).abs()
    image_save_t(diff_warped_one, dir_out, 'diff_warped_one.png', log=True)

def find_depth_feat(t_images, dir_out):
    ''' find sparse feat
        classify sparse feat
        match sparse feat
    '''
    size = list(t_images[0]['img'].size()[2:])
    num_ch = t_images[0]['img'].size(1)

    img = t_images[0]['img']

    skernel = make_gauss_kernel(num_ch, 9)
    img = get_blur(img, skernel)

    gkernel = make_grad_kernel(num_ch)
    grad_img = get_grad_map(img, gkernel)
    grad_img_l = [0]*num_ch
    for c in range(num_ch):
        grad_img_l[c] = (grad_img[0][2*c].pow(2) + grad_img[0][2*c+1].pow(2)).sqrt().unsqueeze(0)
    grad_img1 = torch.cat(grad_img_l, 0).unsqueeze(0)
# print(grad_img1.size())

    for i in range(grad_img1.size(1)):
        image_save_t(grad_img1.narrow(1, i, 1), dir_out, 'grad_'+str(i)+'.png', log=True)

    greys = []
    greys_grads = []
    greys_grad1 = []
    for i in range(len(t_images)):
        grey = t_images[i]['img'].mean(1, keepdim=True)
        sk = make_gauss_kernel(1, 5)
        grey = get_blur(grey, sk)
        greys.append(grey)

        gk = make_grad_kernel(1)
        grad_grey = get_grad_map(grey, gk)
        greys_grad1.append(grad_grey)
        grad_grey1 = (grad_grey[0][0].pow(2) + grad_grey[0][1].pow(2)).sqrt().unsqueeze(0).unsqueeze(0)
        image_save_t(grad_grey1, dir_out, 'grad_grey_'+str(i)+'.png', log=True)
        greys_grads.append(grad_grey1)

    # image -> universal features ->
    # from universal feat get compositional feat w/min collisions for same image and stable under deformations
    size = greys[0].size()[2:]
    grid = make_grid(size, log=True)
# print(grid[0, 0:3, 0:3, 1])
    target = grid.permute([0, 3, 1, 2])
    lr = 0.01
    ksize = 19
    num_steps = 1000
    numf = 4
    inp_img0 = torch.cat([greys[0], greys_grad1[0], greys_grads[0]], 1) #.requires_grad_()
    inp_mean = inp_img0.mean([2, 3], keepdim=True)
    inp_img = inp_img0 - inp_mean
# tar_mean = target.mean([2, 3], keepdim=True)
# print(tar_mean[0].min(dim=0)[0].item(), tar_mean[0].max(dim=0)[0].item())
    print(inp_img.size())
# inp = inp_img.view(inp_img.size(1), -1)
    ksize2 = ksize**2
    krn = torch.Tensor(inp_img.size(1)*ksize2, inp_img.size(1), ksize2).zero_()
    count = 0
    for i in range(inp_img.size(1)):
        for j in range(ksize2):
            krn[count][i][j] = 1
            count += 1
    krn = krn.view(inp_img.size(1)*ksize2, inp_img.size(1), ksize, ksize)
    inp = F.conv2d(inp_img, krn, padding=(ksize-1)//2)
    print(inp.size())
    inp = inp.view(inp.size(1), -1)
    A = inp.matmul(inp.t())
    B = inp.matmul(target.view(2, -1).t())
    C = A.pinverse().matmul(B)
    weight = C.t().view(2, inp_img.size(1), ksize, ksize)
    print(A.size(), B.size(), C.size(), weight.size())
    pred = F.conv2d(inp_img, weight, padding=(ksize-1)//2)
    loss = (pred - target).pow(2).mean(1, keepdim=True)
    loss_mean = loss.mean()
    print('loss=', loss_mean.item(), 'loss min,max, med:', loss.min().item(), loss.max().item(), loss.median().item())
    loss_map = loss.lt(0.1).float()
    print(loss_map.size(), loss_map.min().item(), loss_map.max().item())
    print('sum loss thresh:', loss_map.sum().item())
    image_save_t(loss_map, dir_out, 'loss_map.png', log=True)
    image_save_t(loss, dir_out, 'loss_map_f.png', log=True)



def main():

    dirname = "/repos/data1/"
    dir_out = '/repos/data_out/'

    files = get_files(dirname, log=True)

    if False:
        # np image test
        resize = (300, 400)
        images = load_all_images_np(files, resize)
        dimg = images[0]['img'] - images[1]['img']
        image_save_np(np.abs(dimg), dir_out, 'dimg.png', log=True)

        simg = (np.array(images[0]['img']) + np.array(images[1]['img']))*0.5
        image_save_np(simg, dir_out, 'simg.png', log=True)

    # torch image test
    resize = (400, 300)
    t_images = load_all_images_t(files, resize)

    diff = t_images[0]['img'] - t_images[1]['img']
    aver = (t_images[0]['img'] + t_images[1]['img'])*0.5

    image_save_t(diff.abs(), dir_out, 'diff.png', log=True)
    image_save_t(aver, dir_out, 'aver.png', log=True)

# find depth
# find_depth_rgb(t_images, dir_out)

    find_depth_feat(t_images, dir_out)



if __name__ == '__main__':
    main()
