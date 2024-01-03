import torch
import torch.nn as nn
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def avg_heads(cam, grad=None):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    if grad != None:
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0) # filter negative

    return cam


def grad_rollout(attentions, gradients, discard_ratio,vis_scale='ss',level=3, learnable_weights=False):
    if vis_scale == 'ss':

        result = torch.eye(attentions[0].size(-1))
        # The order of obtaining gradients and attention scores is reversed
        gradients = gradients[::-1]
        with torch.no_grad():
            for attention, grad in zip(attentions, gradients):                
                weights = grad
                attention_heads_fused = (attention*weights).clamp(min=0).mean(axis=1)

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                #a = (attention_heads_fused)/2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)
        
        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0,0,1:]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        print(mask)
        return mask
    else:
        mask_all = []
        '''
        attentions: [transformer_20x.layer_0,transformer_20x.layer_1,transformer_20x.layer_2,
                    transformer_10x.layer_0,transformer_10x.layer_1,transformer_10x.layer_2,
                    transformer_5x.layer_0,transformer_5x.layer_1,transformer_5x.layer_2]
        '''
        if learnable_weights:
            w = attentions[-1]
            w = torch.softmax(w,dim=1)
            w = w.detach().numpy()
            #print(w)
            attns = attentions[:-1]
            grads = gradients[1:]
        else:
            attns = attentions
            grads = gradients
        grads = grads[::-1]
        with torch.no_grad():
            for i in range(3):
                attns_curl = attns[level*i:level*(i+1)]
                grads_curl = grads[level*i:level*(i+1)]
                result = torch.eye(attns_curl[0].size(-1))
                for attn,grad in zip(attns_curl,grads_curl):
                    weights = grad

                    attention_heads_fused = (attn*grad).clamp(min=0).mean(axis=1)


                    flat = attention_heads_fused.view(attention_heads_fused.size(0),-1)
                    _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                    flat[0, indices] = 0
                    I = torch.eye(attention_heads_fused.size(-1))
                    a = (attention_heads_fused + 1.0*I)/2
                    #a = (attention_heads_fused)/2
                    a = a / a.sum(dim=-1)
                    result = torch.matmul(a, result)
                mask = result[0,0,1:]
                width = int(mask.size(-1)**0.5)
                mask = mask.reshape(width, width).numpy()
                mask = mask / np.max(mask)

                mask_all.append(mask)
            
        if learnable_weights: return mask_all, w
        return mask_all

def grad_cam(attentions, gradients, vis_scale, level, learnable_weights=False):
    if vis_scale == 'ss':
        print(attentions[-1].shape)
        gradients = gradients[::-1]
        #print(gradients[0].shape)
        with torch.no_grad():
            attn = attentions[-1] # h,s,s
            grad = gradients[-1] # h,s,s

            print(attn.shape)

            
            attn = attn[0,:,0,1:].reshape((-1,int((attn.shape[-1]-1)**0.5),int((attn.shape[-1]-1)**0.5)))
            grad = grad[0,:,0,1:].reshape((-1,int((gradients[-1].shape[-1]-1)**0.5),int((gradients[-1].shape[-1]-1)**0.5)))

            # h,n,n
            cam_grad = (grad*attn).mean(0).clamp(min=0) #n,n
            cam_grad = (cam_grad-cam_grad.min())/(cam_grad.max()-cam_grad.min())
            print(cam_grad)
            

        return cam_grad.numpy()
    else:
        ## multi-scale
        cam_grad_all = []
        if learnable_weights:
            w = attentions[-1]
            w = torch.softmax(w,dim=1)
            w = w.detach().numpy()
            #print(w)
            attns = attentions[:-1]
            grads = gradients[1:]
        else:
            attns = attentions
            grads = gradients
        grads = grads[::-1]
        with torch.no_grad():
            for i in range(3):
                #print(f'attns_len:{len(attns)}')
                attn_mag = attns[level*(i+1)-1]
                grad_mag = grads[level*(i+1)-1]

                print(attn_mag.shape)
                attn = attn_mag[0,:,0,1:].reshape((-1,int((attn_mag.shape[-1]-1)**0.5),int((attn_mag.shape[-1]-1)**0.5)))
                grad = grad_mag[0,:,0,1:].reshape((-1,int((grad_mag.shape[-1]-1)**0.5),int((grad_mag.shape[-1]-1)**0.5)))

                # h,n,n
                cam_grad = (grad*attn).mean(0).clamp(min=0) #n,n
                cam_grad = (cam_grad-cam_grad.min())/(cam_grad.max()-cam_grad.min())
                
                print(cam_grad)
                print(cam_grad.shape)
                cam_grad_all.append(cam_grad.numpy())
        
        if learnable_weights: 
            return cam_grad_all, w
        return cam_grad_all


class VITAttentionGradRollout:
    def __init__(self, model, level, 
                attention_layer_name='attend',
                discard_ratio=0.9, 
                vis_type = 'grad_rollout', 
                vis_scale='ms', 
                learnable_weights=False):
        '''
        ROI-level visualization. generate attention heatmap with self-attention matrix of Transformer
        
        args:
            model: ROAM model for drawing visualization heatmap
            level: depth of Transformer block
            discard_ratio: proportion of discarded low attention scores. focus only on the top attentions
            vis_type: type of visualization method. 'grad_rollout' or 'grad_cam'
                grad_cam: only focus on the last layer of Transformer at each magnification level
                grad_rollou: consider all self-attention layers
            vis_scale" single scale (ss) or multi-scale (ms)
                'ss': only compute heatmap at 20x magnification scale
            learnable_weight: whether weight coefficients of each scale in the model are learnable
                'True': obtain the final weights from the model's state dict
                'False': fixed weight coefficients can be obtained according to initial config
        '''
        self.model = model
        self.discard_ratio = discard_ratio
        self.vis_type = vis_type
        self.vis_scale = vis_scale
        self.level = level
        self.learnable_weights = learnable_weights
        
        if self.vis_scale == 'ms':
            att_layer_name = [f'transformer_{s}.layers.{l}.0.fn.attend' for s in [20,10,5] for l in range(level)]
            if learnable_weights:
                att_layer_name += [f'ms_attn.{level}']

            cur_l = 0
            for name, module in self.model.named_modules():
                if att_layer_name[cur_l] in name:
                    module.register_forward_hook(self.get_attention)
                    module.register_backward_hook(self.get_attention_gradient)

                    cur_l += 1
                    if cur_l >= len(att_layer_name):
                        break

        else:
            # the attention scores of transformer20 are only ones needed
            att_layer_name = [f'transformer_{s}.layers.{l}' for s in [20,10,5] for l in range(level)]

            cur_l = 0
            for name, module in self.model.named_modules():
                if attention_layer_name in name and att_layer_name[cur_l] in name:
                    module.register_forward_hook(self.get_attention)

                    module.register_backward_hook(self.get_attention_gradient)
                    print(name,'is attention')
                    cur_l += 1
                    if cur_l >= level:
                        break
                #print(name)
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())
    
    # def save_attn_gradients(self, attn_gradients):
    #     print('grad_hook')
    #     print(attn_gradients[0,0,0,1:10])
    #     self.attn_gradients = attn_gradients

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        _,output = self.model(input_tensor.unsqueeze(0),vis_mode=3)
        #print(output.shape)
        loss_fn = nn.CrossEntropyLoss()

        category_mask = torch.zeros(output.size()).cuda()
        category_mask[:, category_index] = 1


        loss = (output*category_mask).sum()

        loss.backward()

        #print(self.vis_type)
        
        if self.vis_type == 'grad_rollout':
            return grad_rollout(self.attentions, self.attention_gradients,
                self.discard_ratio, self.vis_scale,self.level,self.learnable_weights)
        else:
            ## grad_cam
            return grad_cam(self.attentions, self.attention_gradients, self.vis_scale, self.level, self.learnable_weights)
