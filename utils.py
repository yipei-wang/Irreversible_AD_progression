#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import torch
import random
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models
from typing import Type, Any, Callable, Union, List, Optional
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
from captum import attr
from collections import deque

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.benchmark import get_example_data, plot_example


# image_size = 128
# edge = 8
# grid = edge**2
# patch_size = image_size//edge


# In[2]:


def denorm(x, device = None):
    
    if x.shape[-3] == 3:
        if device == None:
            device = x.device
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx[:, 0, :, :] = xx[:, 0, :, :] * std[0] + mean[0]
        xx[:, 1, :, :] = xx[:, 1, :, :] * std[1] + mean[1]
        xx[:, 2, :, :] = xx[:, 2, :, :] * std[2] + mean[2]
    else:
        if device == None:
            device = x.device
        mean = 0.1307
        std = 0.3081
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx = xx * std + mean
        
    return xx
    
    
def imshow(img,
           nrow = 10,
           figsize = (10, 10),
           save = [False, None]):
#     npimg = img.numpy()
    npimg = torchvision.utils.make_grid(img.cpu().detach(), nrow = nrow)
    plt.figure(figsize = figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if save[0]:
        plt.savefig(save[1], bbox_inches='tight')
    plt.show()
    
def normalize(x):
    if x.shape[-3] == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if len(x.shape) == 4:
            xx = x.clone().detach().to(x.device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(x.device)

        xx[:, 0, :, :] = (xx[:, 0, :, :] - mean[0]) / std[0]
        xx[:, 1, :, :] = (xx[:, 1, :, :] - mean[1]) / std[1]
        xx[:, 2, :, :] = (xx[:, 2, :, :] - mean[2]) / std[2]
    else:
        mean = 0.1307
        std = 0.3081
        if len(x.shape) == 4:
            xx = x.clone().detach().to(x.device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(x.device)

        xx = (xx - mean) / std
        
    return xx.squeeze()

def get_n_params(model):
    n_param = 0
    for name, param in model.named_parameters():
        n_param += torch.tensor(param.shape).prod()

    print(f'Then number of parameters is {n_param}.')
    
    return n_param


# In[4]:


## Masking functions

def masking(image, reference, idx = 0, edge = 8):
    '''
    image:3 x image_size x image_size
    '''
    channel = image.shape[-3]
    image_size = image.shape[-1]
    grid = edge**2
    patch_size = image_size//edge
    
    assert idx >= 0 and idx <grid, 'idx out of range'
    x = idx //edge
    y = idx % edge
    masked = image.detach().clone()
    if reference == 'zero':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = 0
    elif reference == 'random':
        reference_values = normalize(torch.rand(channel,patch_size, patch_size).to(image.device)).squeeze()
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = reference_values
    elif reference == 'mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = image.mean()
    elif reference == 'patch_mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
        image[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size].mean()
    elif reference == 'channel_mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
        image.mean(dim=[1,2]).view(3,1,1).expand(3,patch_size,patch_size)
        
    return masked

def greedy(model, image, label, follow_prob = True, mode = 'top', reference = 'zero', edge = 8):
    '''
    image: 3 x image_size x image_size
    label: int/long
    '''  
    channel = image.shape[-3]
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise ValueError("only 1 image at a time")

    image_size = image.shape[-1]
    grid = edge**2
    patch_size = image_size//edge
    
    all_idx = set(range(grid))
    traj = []
    Pred = []
    Prob = []
    model.eval()
    with torch.no_grad():
        new_image = image.detach().clone()
        pred = model(new_image[None])[:, label].item()
        prob = torch.softmax(model(new_image[None]), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        while len(all_idx) > 1: 
            masked = []
            ids = []
            for i in all_idx:
                masked.append(masking(new_image, reference, i, edge = edge))
                ids.append(i)
            masked = torch.stack(masked)
            pred = model(masked)[:, label]
            prob = torch.softmax(model(masked), dim = 1)[:, label]
            
            if mode == 'bottom':
                if follow_prob:
                    traj.append(ids[prob.argmax().item()])
                    all_idx.remove(traj[-1])
                    Prob.append(prob.max().item())
                    Pred.append(pred[prob.argmax().item()].item())
                    new_image = masked[prob.argmax().item()].detach().clone()
                else:
                    traj.append(ids[pred.argmax().item()])
                    all_idx.remove(traj[-1])
                    Pred.append(pred.max().item())
                    Prob.append(prob[pred.argmax().item()].item())
                    new_image = masked[pred.argmax().item()].detach().clone()
            elif mode == 'top':
                if follow_prob:
                    traj.append(ids[prob.argmin().item()])
                    all_idx.remove(traj[-1])
                    Prob.append(prob.min().item())
                    Pred.append(pred[prob.argmin().item()].item())
                    new_image = masked[prob.argmin().item()].detach().clone()
                else:
                    traj.append(ids[pred.argmin().item()])
                    all_idx.remove(traj[-1])
                    Pred.append(pred.min().item())
                    Prob.append(prob[pred.argmin().item()].item())
                    new_image = masked[pred.argmin().item()].detach().clone()
            else:
                raise ValueError("mode has to be either 'top' or 'bottom'")  
                
        image_size = image.shape[-1]
                
        if reference == 'zero':
            masked = torch.zeros(1,channel,image_size,image_size).to(image.device)
        elif reference == 'random':
            masked = normalize(torch.rand(1,channel,image_size,image_size).to(image.device)).view(1,channel,image_size,image_size)
        pred = model(masked)[:, label].item()
        prob = torch.softmax(model(masked), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        traj.append(all_idx.pop())
    return traj, Pred, Prob


def demon(image, traj, nrow, figsize = (15, 15), edge = 8):
    channel = image.shape[-3]
    image_size = image.shape[-1]
    masked = image.detach().clone().view(channel,image_size,image_size)
    toshow = [masked[None]]
    for ids in traj:
        masked = masking(masked, 'zero', ids, edge = edge)
        toshow.append(masked[None])
    toshow = torch.cat(toshow)
    imshow(denorm(toshow), nrow = nrow, figsize = figsize)


# In[7]:


## Insertion functions

def insertion(reference_values, image, idx = 0):
    '''
    image:3ximage_sizeximage_size
    '''
    assert idx >= 0 and idx <grid, 'idx out of range'
    x = idx //edge
    y = idx % edge
    inserted = reference_values.detach().clone()
    inserted[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size]        = image[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size]
    return inserted

def greedy_insertion(model, image, label, follow_prob = True, mode = 'top', reference = 'zero'):
    '''
    image: 3 x image_size x image_size
    label: int/long
    '''  
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise ValueError("only 1 image at a time")
    
    all_idx = set(range(grid))
    traj = []
    Pred = []
    Prob = []
    model.eval()
    with torch.no_grad():
        if reference == 'zero':
            reference_values = torch.zeros(image.shape).to(image.device)
        elif reference == 'random':
            reference_values = normalize(torch.rand(image.shape)).to(image.device)         
        pred = model(reference_values[None])[:, label].item()
        prob = torch.softmax(model(reference_values[None]), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        
        while len(all_idx) > 1: 
            inserted = []
            ids = []
            for i in all_idx:
                inserted.append(insertion(reference_values, image, i))
                ids.append(i)
            inserted = torch.stack(inserted).to(image.device)
            pred = model(inserted)[:, label]
            prob = torch.softmax(model(inserted), dim = 1)[:, label]
            
            if mode == 'top':
                if follow_prob:
                    traj.append(ids[prob.argmax().item()])
                    all_idx.remove(traj[-1])
                    Prob.append(prob.max().item())
                    Pred.append(pred[prob.argmax().item()].item())
                    reference_values = inserted[prob.argmax().item()].detach().clone()
                else:
                    traj.append(ids[pred.argmax().item()])
                    all_idx.remove(traj[-1])
                    Pred.append(pred.max().item())
                    Prob.append(prob[pred.argmax().item()].item())
                    reference_values = inserted[pred.argmax().item()].detach().clone()
                    
            elif mode == 'bottom':
                if follow_prob:
                    traj.append(ids[prob.argmin().item()])
                    all_idx.remove(traj[-1])
                    Prob.append(prob.min().item())
                    Pred.append(pred[prob.argmin().item()].item())
                    reference_values = inserted[prob.argmin().item()].detach().clone()

                else:
                    traj.append(ids[pred.argmin().item()])
                    all_idx.remove(traj[-1])
                    Pred.append(pred.min().item())
                    Prob.append(prob[pred.argmin().item()].item())
                    reference_values = inserted[pred.argmin().item()].detach().clone()
            else:
                raise ValueError("mode has to be either 'top' or 'bottom'")  
                
        inserted = image.detach().clone()[None]
        pred = model(inserted)[:, label].item()
        prob = torch.softmax(model(inserted), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        traj.append(all_idx.pop())
    return traj, Pred, Prob


def demon_insertion(image, traj, nrow, figsize = (15, 15)):
    inserted = torch.zeros(image.shape).to(image.device)
#     masked = image.detach().clone().view(3,image_size,image_size)
    toshow = [inserted[None]]
    for ids in traj:
        inserted = insertion(inserted, image, ids)
        toshow.append(inserted[None])
    toshow = torch.cat(toshow)
    imshow(denorm(toshow), nrow = nrow, figsize = figsize)


# In[5]:


## Demnostration functions

def attribute(model, image, label, follow_prob = True, mode = 'top', reference = 'zero', grid = 64, insertion = False):
    image_size = image.shape[-1]
    if not insertion:
        traj, Pred, Prob = greedy(model, image, label, follow_prob, mode, reference)
        saliency = torch.zeros(1, grid).to(image.device)
        for i in range(grid):
            if follow_prob:
                saliency[0, traj[i]] = -Prob[i+1] + Prob[i]
            else:
                saliency[0, traj[i]] = -Pred[i+1] + Pred[i]
        saliency = saliency.view(1,1,edge,edge)
        heatmap = F.interpolate(saliency, size = (image_size, image_size), mode = 'bilinear', align_corners = True)
        return saliency, heatmap
    else:
        traj, Pred, Prob = greedy_insertion(model, image, label, follow_prob, mode, reference)
        saliency = torch.zeros(1, grid).to(image.device)
        for i in range(grid):
            if follow_prob:
                saliency[0, traj[i]] = -Prob[i+1] + Prob[i]
            else:
                saliency[0, traj[i]] = -Pred[i+1] + Pred[i]
        heatmap = F.interpolate(saliency, size = (image_size, image_size), mode = 'bilinear', align_corners = True)
        return saliency, heatmap
    
def saliency_to_traj(saliency, mode = 'bottom'):
    if mode == 'top':
        return np.array(list(reversed(saliency.flatten().argsort()).detach().cpu().numpy()))
    elif mode == 'bottom':
        return np.array(list(saliency.flatten().argsort().detach().cpu().numpy()))
    
def traj_masking(model, image, label, traj, reference = 'zero', edge = 8):
    model.eval()
    masked = image.detach().clone()
    with torch.no_grad():
        inputs = [masked[None]]
        for ids in traj:
            masked = masking(masked, reference, ids, edge = edge)
            masked = masked
            inputs.append(masked[None])
        inputs = torch.cat(inputs).to(image.device)
        pred = model(inputs)
        prob = torch.softmax(pred, dim = 1)
        return pred[:, label].detach().cpu().numpy(), prob[:, label].detach().cpu().numpy(), inputs
#         return pred[:, label].detach(), prob[:, label].detach(), inputs
    
def hook_gc(model, image, label, use_relu = False):
    
    image_size = image.shape[-1]
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output#.detach()
        return hook
    handle = model.layer3.register_forward_hook(get_activation('feature'))
    
    input = image.view(1,3,image_size, image_size).to(image.device).requires_grad_(True)
    output = model(input)
    handle.remove()

    grad = torch.autograd.grad(
        inputs = activation['feature'],
        outputs = output[range(len(output)), label].sum(),
        create_graph = True
    )[0]

    return (activation['feature'].squeeze() * grad.mean(dim = [2,3]).view(1, 256, 1, 1)).sum(dim = 1, keepdim = True)


# In[6]:


## Randomization functions

def greedy_random(model, image, label, follow_prob = True, mode = 'top', reference = 'zero', K = 3, edge = 8):
    '''
    image: 3 x image_size x image_size
    label: int/long
    '''  
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise ValueError("only 1 image at a time")
    channel = image.shape[-3]
    image_size = image.shape[-1]
    grid = edge**2
    
    all_idx = set(range(grid))
    traj = []
    Pred = []
    Prob = []
    model.eval()
    with torch.no_grad():
        new_image = image.detach().clone()
        pred = model(new_image[None])[:, label].item()
        prob = torch.softmax(model(new_image[None]), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        while len(all_idx) > 1: 
            masked = []
            ids = []
            for i in all_idx:
                masked.append(masking(new_image, reference, i, edge = edge))
                ids.append(i)
            masked = torch.stack(masked)
            pred = model(masked)[:, label]
            prob = torch.softmax(model(masked), dim = 1)[:, label]
            
            if mode == 'bottom':
                if follow_prob:
                    k = min(len(all_idx), K)
                    selected = random.choice(prob.sort()[1][-k:]).item()
                    traj.append(ids[selected])
                    all_idx.remove(traj[-1])
                    Pred.append(pred[selected].item())
                    Prob.append(prob[selected].item())
                    new_image = masked[selected].detach().clone()
                else:
                    k = min(len(all_idx), K)
                    selected = random.choice(pred.sort()[1][-k:]).item()
                    traj.append(ids[selected])
                    all_idx.remove(traj[-1])
                    Pred.append(pred[selected].item())
                    Prob.append(prob[selected].item())
                    new_image = masked[selected].detach().clone()
            elif mode == 'top':
                if follow_prob:
                    k = min(len(all_idx), K)
                    selected = random.choice(prob.sort()[1][:k]).item()
                    traj.append(ids[selected])
                    all_idx.remove(traj[-1])
                    Pred.append(pred[selected].item())
                    Prob.append(prob[selected].item())
                    new_image = masked[selected].detach().clone()
                else:
                    k = min(len(all_idx), K)
                    selected = random.choice(pred.sort()[1][:k]).item()
                    traj.append(ids[selected])
                    all_idx.remove(traj[-1])
                    Pred.append(pred[selected].item())
                    Prob.append(prob[selected].item())
                    new_image = masked[selected].detach().clone()
            else:
                raise ValueError("mode has to be either 'top' or 'bottom'")  
        image_size = image.shape[-1]
                
        if reference == 'zero':
            masked = torch.zeros(1,channel,image_size,image_size).to(image.device)
        elif reference == 'random':
            masked = normalize(torch.rand(1,channel,image_size,image_size).to(image.device)).view(1,channel,image_size,image_size)
        pred = model(masked)[:, label].item()
        prob = torch.softmax(model(masked), dim = 1)[:, label].item()
        Pred.append(pred)
        Prob.append(prob)
        traj.append(all_idx.pop())
    return traj, Pred, Prob

def best_random(model, image, label, follow_prob = False, reference = 'zero', mode = 'bottom', N = 5, K = 3, edge = 8):
    
    traj_best, Pred_best, Prob_best = greedy_random(model, image, label, False, mode = mode,
                                                    reference = reference, K = K, edge = edge)
    Pred_best_reversed, Prob_best_reversed, _ = traj_masking(model, image, label, reversed(traj_best), edge = edge)
    
    for n in range(N-1):
        traj_random, Pred_random, Prob_random = greedy_random(model, image, label, False, mode = mode,
                                                              reference = reference, K = K, edge = edge)
        Pred_random_reversed, Prob_random_reversed, _ = traj_masking(model, image, label, reversed(traj_random), edge = edge)
        
        if follow_prob:
            if sum(Prob_best) - sum(Prob_best_reversed) < sum(Prob_random) - sum(Prob_random_reversed):
                Prob_best = Prob_random.copy()
                Prob_best_reversed = Prob_random_reversed.copy()
                traj_best = traj_random.copy()
        else:
            if sum(Pred_best) - sum(Pred_best_reversed) < sum(Pred_random) - sum(Pred_random_reversed):
                Pred_best = Pred_random.copy()
                Pred_best_reversed = Pred_random_reversed.copy()
                traj_best = traj_random.copy()
    if follow_prob:
        return Prob_best, Prob_best_reversed, traj_best
    else:
        return Pred_best, Pred_best_reversed, traj_best

def many_random(model, image, label, follow_prob = False, reference = 'zero', mode = 'bottom', N = 5, K = 3, edge = 8):
    
    Traj = []
    Pred = []
    Pred_re = []
    
    for n in tqdm(range(N)):
        traj, pred, prob = greedy_random(model, image, label, False, mode = mode,
                                                      reference = reference, K = K)
        pred_re, probbest, _ = traj_masking(model, image, label, reversed(traj), edge = edge)
        
        Traj.append(traj)
        Pred.append(pred)
        Pred_re.append(pred_re)
        
    return np.array(Pred), np.array(Pred_re), np.array(Traj)


# In[ ]:




