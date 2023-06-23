import torch
import numpy as np
import math
import copy


# builder = Builder(arch=args.arch, pretrained_weight=pretrained, model=model, remain_filter_indices=remain_filters, cuda=args.cuda, labels=labels)


class Builder(object):
    def __init__(self, arch, pretrained_weight, model, remain_filter_indices, cuda, labels) -> None:
        self.arch = arch
        self.model = model
        self.param_dict = pretrained_weight
        self.remain_filter_indices = remain_filter_indices

        self.labels = labels

        self.decompose_weight = None
        self.cuda = True if cuda =='cuda' else False

    def bulid_model(self):
        z = None

        # decompose_weight : pruning한 parameter matrix 저장
        self.decompose_weight = list(self.param_dict.values())

        layer_id = -1

        dense_layer_id = 0
        dense_channel_index = [i for i in range(24)] 
        # 파라미터의 개수만큼 반복문이 돌아간다. 
        for index, layer in enumerate(self.param_dict):

            # 매 for문 마다 orginal은 original parameter값을 가진다. 
            original = self.param_dict[layer]

            if self.arch == 'vgg':

                if 'feature' in layer:
                    

                    # param_dict[layer]가 뭐지? 
                    # 파라미터의 차원수가 4인 weight만 pruning 한다. 
                    if len(self.param_dict[layer].shape) == 4:

                        layer_id += 1

                        output_channel_index = self.remain_filter_indices[layer_id]

                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o


                        # np.array(output_channel_index): 각 레이어의 남아있는 필터의 인덱스
                        # output: scailing matrix 리턴
                        x = self.create_scaling_mat_conv_thres_bn(
                                                                self.param_dict[layer].cpu().detach().numpy(), 
                                                                np.array(output_channel_index)
                                                                )
                        # z는 데이터 type이 float인 x이다. 
                        z = torch.from_numpy(x).type(dtype=torch.float) 
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = original[output_channel_index,:,:,:]

                        input_channel_index = output_channel_index

                        self.decompose_weight[index] = pruned

                    elif len(self.param_dict[layer].shape):
                        
                        pruned = self.param_dict[layer][input_channel_index]

                        self.decompose_weight[index] = pruned

                else:
                    pruned = torch.zeros(original.shape[0],z.shape[0])

                    if self.cuda:
                        pruned = pruned.cuda()

                    for i, f in enumerate(original):
                        o_old = f.view(z.shape[1],-1)
                        o = torch.mm(z,o_old).view(-1)
                        pruned[i,:] = o

                    self.decompose_weight[index] = pruned

                    break

            elif 'resnet' in self.arch and '50' not in self.arch:
                if 'layer' in layer : 
                    
                    # 하나의 basic block 시작에 레이어수 + 1
                    if 'conv1.weight' in layer: 
                        layer_id += 1

                    if 'conv1' in layer:

                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = original[output_channel_index,:,:,:] # (n, c, h, w) = (15, 16, 3, 3)
                        
                        input_channel_index = output_channel_index

                        self.decompose_weight[index] = pruned

                    elif 'bn1' in layer :
                        # 배열의 차원 수가 0이 아닌 경우 
                        if len(self.param_dict[layer].shape):

                            pruned = self.param_dict[layer][input_channel_index]
                            
                            self.decompose_weight[index] = pruned # (15)
                    
                    elif 'conv2' in layer :

                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]): #f.shape (16, 3, 3) # self.param_dict[layer] (16, 16, 3, 3)
                                o = f.view(f.shape[0],-1) # (16, 9) # 가중치를 2차원으로 변환 # 채널 차원을 유지하고 나머지 차원을 하나로 평면화
                                o = torch.mm(z,o) # z: (15, 16)  # o : (16, 9)  # 최종 o:(15, 9)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2]) # (15, 3 ,3)
                                original[i,:,:,:] = o
                        
                        scaled = original

                        self.decompose_weight[index] = scaled

            elif 'resnet' in self.arch and '50' in self.arch:
                if 'layer' in layer : 

                    if 'conv1.weight' in layer or 'conv2.weight' in layer: 
                        layer_id += 1

                    if 'conv1' in layer:

                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = original[output_channel_index,:,:,:]

                        input_channel_index = output_channel_index

                        self.decompose_weight[index] = pruned

                    elif 'bn1' in layer:

                        if len(self.param_dict[layer].shape):

                            pruned = self.param_dict[layer][input_channel_index]

                            self.decompose_weight[index] = pruned
                    
                    elif 'conv2' in layer :
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = scaled[output_channel_index,:,:,:]

                        input_channel_index = output_channel_index

                        self.decompose_weight[index] = pruned

                    elif 'bn2' in layer:
                        if len(self.param_dict[layer].shape):

                            pruned = self.param_dict[layer][input_channel_index]

                            self.decompose_weight[index] = pruned

                    elif 'conv3' in layer:
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        self.decompose_weight[index] = scaled
        
            elif self.arch == 'densenet40':
                if 'dense' in layer:
                    if len(self.param_dict[layer].shape) == 4:

                        layer_id += 1

                        output_channel_index = self.remain_filter_indices[layer_id]

                        if z != None:
                            original = original[:,dense_channel_index,:,:]

                        x = self.create_scaling_mat_conv_thres_bn(
                                                                self.param_dict[layer].cpu().detach().numpy(),
                                                                np.array(output_channel_index)
                                                                )

                        z = torch.from_numpy(x).type(dtype=torch.float)

                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = original[output_channel_index,:,:,:]

                        output_channel_index_d = np.array(copy.deepcopy(output_channel_index))
                        output_channel_index_d = output_channel_index_d + 12*(layer_id) + 24
                        dense_channel_index.extend(output_channel_index_d)

                        self.decompose_weight[index] = pruned

                    elif len(self.param_dict[layer].shape) and "dense1.0.bn" not in layer:

                        pruned = self.param_dict[layer][dense_channel_index]

                        self.decompose_weight[index] = pruned
                elif 'trans' in layer:
                    if len(self.param_dict[layer].shape) == 4:
                        pruned = original[:,dense_channel_index,:,:]
                        dense_layer_id += 1
                        dense_channel_index = [i for i in range(12*12*dense_layer_id +24)]
                        self.decompose_weight[index] = pruned
                    elif len(self.param_dict[layer].shape):

                        pruned = self.param_dict[layer][dense_channel_index]

                        self.decompose_weight[index] = pruned

                elif 'bn' in layer:
                    if len(self.param_dict[layer].shape):
                        pruned = self.param_dict[layer][dense_channel_index]
                        if self.cuda:
                            pruned = pruned.cuda()

                        self.decompose_weight[index] = pruned
                elif 'fc' in layer:
                    pruned = original[:,dense_channel_index]

                    if self.cuda:
                        pruned = pruned.cuda()

                    self.decompose_weight[index] = pruned

                    break

            elif self.arch == 'mobilenet':
                    
                    # 첫번째 conv
                    if 'conv.0.0.weight' in layer:

                        layer_id += 1

                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        pruned = original[output_channel_index,:,:,:]
                        
                        input_channel_index = output_channel_index

                        self.decompose_weight[index] = pruned

                    
                    # 배열의 차원 수가 0이 아닌 경우 # bn layer
                    elif 'conv.0.1.weight' in layer:
                        if len(self.param_dict[layer].shape):

                            pruned = self.param_dict[layer][input_channel_index]
                            
                            self.decompose_weight[index] = pruned
                    

                    # 두번째 conv
                    if 'conv.1.0.weight' in layer :

                        if z != None:
                            original = original[input_channel_index,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        self.decompose_weight[index] = scaled


        self._weight_init()

    # 리스트에 저장된 가중치를 순서대로 가져와 모델의 각 계층에 대해 가중치를 초기화
    def _weight_init(self):
        for layer in self.model.state_dict():
            decomposed_weight = self.decompose_weight.pop(0)
            self.model.state_dict()[layer].copy_(decomposed_weight)




# x = self.create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(), np.array(output_channel_index))

    def create_scaling_mat_conv_thres_bn(self, weight, ind):
