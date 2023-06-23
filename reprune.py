import os
from os import listdir, makedirs
from os.path import isfile, join
import argparse
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn import cluster, metrics, mixture, preprocessing
from nni.compression.pytorch.utils import count_flops_params
import shutil
from termcolor import colored
from tqdm import tqdm
from build_model import Builder

import models
import util
import math

parser = argparse.ArgumentParser(description="purning")
parser.add_argument('--arch', type=str, default = 'mobilenetv2')
parser.add_argument('--dataset', type=str, default = 'cifar10')
parser.add_argument("--save", type=str, default="./mobilenetv2_cifar10")
parser.add_argument("--prune_ratio", type=float, default=0.5)
parser.add_argument("--datapath", type=str, default="./")
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--ngpu", type=str, default="cuda:0")
parser.add_argument("--gamma", type=float, default=0.5)
#hyper parmeter
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[30, 60])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=256)
parser.add_argument("--finetuning", type=bool, default= True)
parser.add_argument("--criterion", type=str, default="reprune")
parser.add_argument("--selection_with_norm", action='store_true')
parser.add_argument("--replacement", action='store_true')
parser.add_argument("--scaling", action='store_true')
parser.add_argument("--evaluate", action='store_true', default=False)
parser.add_argument("--threshold_type", type=str, default="ward")
parser.add_argument("--linkage_type", type=str, default="ward", help="ward, complete, average, single")

args = parser.parse_args()
print("=> Parameter : {}".format(args))

if args.arch == "vgg":
    model_name = "vgg16"

elif args.arch == "resnet":
    model_name == "resnet56"

else:
    model_name = "mobilenetv2"



if args.finetuning:
    print("Fine tuning")
    if model_name.startswith("vgg"):
        args.pretrained = "./logs/vgg/{}_baseline_{}/model_best.pth.tar".format(model_name,args.dataset)
    elif model_name.startswith("resnet"):
        args.pretrained = "./logs/resnet/{}_baseline_{}/model_best.pth.tar".format(model_name,args.dataset)
    elif model_name.startswith("densenet"):
        args.pretrained = "./logs/densenet/{}_baseline_{}/model_best.pth.tar".format(model_name,args.dataset)
    elif model_name.startswith("mobilenetv2"):
        args.pretrained = "./logs/mobilenetv2/{}_baseline_{}/model_best.pth.tar".format(model_name,args.dataset)

else:
    print("Init")
################################ Check Parser #######################################

################################# train method ########################################
torch.manual_seed(args.seed)
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = args.ngpu if args.cuda else "cpu"
datapath = os.path.join(args.datapath , args.dataset) # ./cifar10
g_epoch = 0

# learning rate 조정하기
def adjust_learning_rate(optimizer, epoch, step, len_iter):
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)


    # 모든 파라미터 그룹에 'lr' parameter 추가     
    for param_group in optimizer.param_groups: # optimizer의 parameter들을 dictionary 형태로 가지고 있음.
        param_group['lr'] = lr
        print(param_group)


def train(model, epoch, optimizer, criterion):
    global g_epoch
    model.train()
    len_iter = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        adjust_learning_rate(optimizer, epoch, batch_idx, len_iter) # lr 설정
        output = model(data)
        loss = criterion(output, target)

        # optimizer의 파라미터들의 변화도(gradient)를 0으로 재 설정 
        optimizer.zero_grad() 

        # optimizer 파라미터들의 변화도(gradient) 저장
        loss.backward()

        # optimizer의 파라미터들 업데이트 
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                  g_epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.data.item()))

        # loss를 가져올 때는 loss.data.item
    g_epoch += 1
    for param_group in optimizer.param_groups:
        print("Learning rate : ", param_group['lr'])

    return loss.data.item()

def accuracy(output, target, topk=(1,)):
    # 역전파를 하지 않고, accuracy만을 계산하는것이므로 torch.no_grad() 선언
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) 

        _, pred = output.topk(maxk, 1, True, True) # output에서 maxk개 만큼, 큰 값(그리고 인덱스)부터 차례대로 출력
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # target과 prediction 값이 맞는지의 유무를 저장

        tot_correct = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            tot_correct.append(correct_k)
        return tot_correct # topk개의 정확도를 나타낸다. 
    
def test(model, ece=False):
    model.eval()
    test_loss = 0
    corr1 = 0
    corr5 = 0
    criterion = nn.CrossEntropyLoss().to(device)
    
    if ece:
        logit_set = np.zeros([10000, 10])
        counter = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if ece:
                for i, per_sample in enumerate(output):
                    logit_set[counter+i] = per_sample.detach().cpu().numpy()
                counter += output.size(0)

            test_loss += criterion(output, target).item()

            corr1_, corr5_ = accuracy(output, target, topk=(1, 5))
            corr1 += corr1_
            corr5 += corr5_

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\nTop-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, corr1.item(), len(test_loader.dataset),
            100. * float(corr1.item() / len(test_loader.dataset)),
            corr5.item(), len(test_loader.dataset),
            100. * float(corr5.item() / len(test_loader.dataset))))
    if ece:
        return float(corr1.item()/len(test_loader.dataset)), float(corr5.item()/len(test_loader.dataset)), test_loss, logit_set
    else:
        return float(corr1.item()/len(test_loader.dataset)), float(corr5.item()/len(test_loader.dataset)), test_loss
def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
################################# train method ########################################

################################# pruning method ########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
# from Ward_converter import Ward_Converter
import math



# output: filter_weight_flattened : flatten된 weight들을 원소로 갖는 리스트
def load_filter_weight(weight_path, weight_num):
    weight_name = str(weight_num) + '.npy'
    # weight_name : "0.npy"

    filter_weight = np.load(join(weight_path,weight_name))
    # filter_weight에 weight 파일들 로드

    #print("filter shape", filter_weight.shape)
    filter_weight_flattened = np.array([each_filter_weight.flatten() for each_filter_weight in filter_weight])
    # filter_weight_flattened : flatten된 weight들을 원소로 갖는 리스트
    return filter_weight_flattened


# mid_distances = cal_distance(mid_filter[n], 3)
# layer = mid_filter[1]인 경우
# layer.shape = (16,27)
# filter_channel = 27
# kernel = 9
# range(0, 27, 9)

# input: layer, 커널 크기
# output: pixel_distance(리스트의 원소는, 해당 레이어의 각 커널 축 별 군집들 사이의 거리))
def cal_distance(layer,kernel_size): 
   
    _, filter_channel = layer.shape
   
    kernel= kernel_size*kernel_size
    pixel_distance = []
    

    for c_ind, c in enumerate(range(0,filter_channel,kernel)):
        pixels = layer[:,c:c+kernel]

        # distance_threshold = 0 : 군집을 병합할때 threshold를 설정하지 않음 
        clustering = cluster.AgglomerativeClustering(linkage = args.linkage_type, n_clusters = None, distance_threshold = 0,compute_distances = True)
        clustering.fit(pixels)
        
        distances = clustering.distances_ # 각 단계에서 병합되는 군집들 사이의 거리, 군집들이 하나로 병합될때까지의 거리를 담고있음. 
        pixel_distance.extend(distances)
        
        
    return np.array(pixel_distance) 

#newpruning(mid_filter[n], n, 3, thr, args.threshold_type)
# layer : layer의 weight가 flatten 된 형태
    # layer_num = 몇번째 레이어인지
    # kernel의 가로 혹은 세로 길이
    # thr : threshold
    # threshold_type = ward
    # layer.shape : filter의 개수 * (filter의 채널 수) 

    # filter_count = filter의 개수
    # filter_channel = filter의 채널 수


#output 
# max_filter_num: 최대 군집 수 (주어진 레이어에서, 여러 커널 축 중 "cluster 개수가 최대가 되는 커널의 cluster 개수")
# filter_label: 리스트의 원소는 '주어진 레이어에서, 커널 축에서의 cluster 라벨들의 모임' 이다. 

def newpruning(layer, layer_num, kerenl_size, thr, threshold_type): 
    
    filter_count, filter_channel = layer.shape
    kernel = kerenl_size*kerenl_size
    max_filter_num =0
  
    for c in range(0, filter_channel,kernel):

        # 레이어의 채널 하나에 해당하는 2차원의 pixel
        pixels = layer[:,c:c+kernel]

        pixel_clustering = cluster.AgglomerativeClustering(linkage = args.linkage_type, n_clusters = None, distance_threshold = thr,compute_distances = True)
        pixel_clustering.fit(pixels)
        num = len(set(pixel_clustering.labels_))
        # num: 한 커널에서의 cluster의 개수( threshold를 설정했을 때, 한 커널에서 몇개의 클러스터가 나오는지 결정됨)
       
        if num > max_filter_num:
            max_filter_num = num

    
        label = np.array(pixel_clustering.labels_).reshape(1,-1).transpose()
        #print('label shape of {}th layer and {}kernel: {}'.format(layer_num, c, label.shape))

        if c == 0:
            filter_label = label 
        else:
            # hstack : 수평으로(가로로, horizontal) 연결
            filter_label = np.hstack((filter_label,label))

    return max_filter_num, filter_label

def detect_bottleneck_label(layer, kerenl_size, cluster_num):
    filter_count, filter_channel = layer.shape
    kernel = kerenl_size*kerenl_size
    pixel_distance = []
    threshold = 0
    max_filter_num =0
    max_distance = 0
    # append kernel distance
    if filter_count == cluster_num:
        threshold = 0
    elif filter_count == 1:
        threshold = 1e10
    else:
        for c in range(0, filter_channel,kernel):
            pixels = layer[:,c:c+kernel]
            pixel_clustering = cluster.AgglomerativeClustering(linkage = args.linkage_type,n_clusters = cluster_num,compute_distances = True)
            pixel_clustering.fit(pixels)
            if max_distance < pixel_clustering.distances_[-cluster_num]:
                max_distance = pixel_clustering.distances_[-cluster_num]
            pixel_distance.extend(pixel_clustering.distances_)

        pixel_distance.sort()
        index = pixel_distance.index(max_distance)
        i = 1
        while pixel_distance[index] == pixel_distance[index+i]:
            i += 1
        threshold = pixel_distance[index+i]
        
    # clustering and return label
    for c in range(0, filter_channel,kernel):
        pixels = layer[:,c:c+kernel]
        pixel_clustering = cluster.AgglomerativeClustering(linkage = args.linkage_type,n_clusters = None, distance_threshold = threshold,compute_distances = True)
        pixel_clustering.fit(pixels)
        num = len(set(pixel_clustering.labels_))
        if num>max_filter_num:
            max_filter_num = num
        label = np.array(pixel_clustering.labels_).reshape(1,-1).transpose()
        if c == 0:
            filter_label = label         
        else:
            filter_label = np.hstack((filter_label,label))
    assert cluster_num == max_filter_num 
    return max_filter_num, filter_label


def cdf(data, thr, threshold_type):
    sorted_random_data = np.sort(data)
    p = 1. * np.arange(len(sorted_random_data)) / float(len(sorted_random_data) - 1)
    return(sorted_random_data[p>=thr][0])

def cprint(print_string, color):
    cmap={"r": "red", "g": "green", "y":"yellow", "b":"blue"}
    print(colored("{}".format(print_string), cmap[color]))
################################# pruning method #######################################

################################# main #################################################
################################# load Dataset #########################################

#####
cprint("###################start load Dataset######################", 'b')
# 지정된 데이터를 로더에 태움
if args.dataset.startswith('cifar'):
    CIFAR = datasets.CIFAR10 if args.dataset=="cifar10" else datasets.CIFAR100
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) 

    kwargs = {'num_workers': 4, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(
        CIFAR(datapath, train=True, download=True, transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        CIFAR(datapath, train=False, download=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

elif args.dataset == "imagenet":
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    traindir = os.path.join(datapath, 'train')
    testdir = os.path.join(datapath, 'val3')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

################################# load Dataset ##########################################

################################# load weight ###########################################
cprint("###################start load weights######################", 'b')
cprint("start loading filter weight","r")

weight_path = os.path.join("./filter_weights",model_name+"_"+args.dataset,"baseline")
# weight_path : ./filter_weights/renet56_cifar10/baseline
weight_names = [f for f in listdir(weight_path) if isfile(join(weight_path, f))]


weight_num_list = [int(weight_name.replace('.npy','')) for weight_name in weight_names]
weight_num_list.sort()
#weight_num_list : [0, 1, ... 56]


# load selected filters
filter_weight_flattened_dict = {}

for weight_num in weight_num_list:
    print(weight_num,"_layer") # 0_layer, ... 56_layer 이렇게 프린트

    # load_filter_weight: flatten 된 filter weight들을 원소로 가지는 리스트(filter weight들은 array형태)
    # filter_weight_flattened_dict : weight_num을 key 값으로, weight를 value로 갖는 딕셔너리
    filter_weight_flattened_dict[weight_num] = load_filter_weight(weight_path, weight_num)
    print("flatten shape : ", filter_weight_flattened_dict[weight_num].shape)



cprint("finish loading filter weight","r")
print("loaded filter weight num :", len(filter_weight_flattened_dict)) # filter weight의 개수: 0부터 56이므로 57개

################################# load weight #############################################

################################# filter pruning ##########################################
cprint("################### end load weights######################", 'b')
cprint("################### filter pruning ######################", 'b')

replace_mapping_table_list=[]
uncovered_ratio = -1


if args.arch.startswith('vgg'):
    cprint("start vgg pruning","r")

    extracted_filter_weight = filter_weight_flattened_dict
    # extract 3*3 conv
    mid_filter = []
    for n in range(len(extracted_filter_weight)):
        mid_filter.append(extracted_filter_weight[n])

    # mid_filter = filter_weight_flattened : flatten된 weight 56개를 원소로 갖는 리스트

    # layer하나 당 하나의 weight를 갖는다. 
    # 레이어 하나에서 파라미터수를 구하는 공식: 커널사이즈 * input의 채널 수 * output의 채널 수(커널의 개수)
    # 필터의 weight가 4차원인거고 필터마다 하나의 weight를 갖는거군.
    print("layer num: ", len(mid_filter)) # layer의 개수 = weight의 개수 



    # calculate kerenl distance
    # mid 3*3 conv 
    total_mid_distances = None

    print("calculate distance threshold")
    # 반복문이 56번 돌아간다. 
    for n in tqdm(range(len(mid_filter))):
        # input: mid_filter[n] : flatten된 weight
        # output: 
        mid_distances = cal_distance(mid_filter[n], 3)

        if total_mid_distances is None:
            total_mid_distances = mid_distances
        else:
            total_mid_distances = np.concatenate((total_mid_distances, mid_distances), axis=0)

    # total_mid_distances : [ 1, 2, 3, ... ]과 같은 1차원 배열

    # input: pruning ratio: 0.5/method:  ward
    # thr p >= thr인 weight들을 원소로 가지는 리스트 반환
    thr = cdf(total_mid_distances,args.prune_ratio, args.threshold_type)
    print("new thr : ", thr)
    # agglomerative clustring
    # mid 3*3conv

    print("filter pruning")
    mid_cfg = []
    labels = []

    
    for n in tqdm(range(len(mid_filter))):
        filter_nums,label = newpruning(mid_filter[n], n, 3, thr, args.threshold_type)
    
        mid_cfg.append(filter_nums)
        labels.append(label)

    layer_num = len(mid_filter)
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []

    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        layer_kernel = mid_filter[l_ind]
        layer_label = labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)

        coverage_points_num.append(accumulated_index-1)

    # Maximum coverage
    print("Maximum Coverage")
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    # Code book (cluster_num, mid_filters, labels)
    codebook_list = []
    remain_filters = []
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0
    node_used_count = []
    uncovered_cluster_indice = []

    # Maximum coverage
    print("Maximum Coverage")
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector

    for l_ind in range(layer_num):
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        remain_filters.append([])
        
        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            filter = selector.get_filter()

            node_used_count[l_ind][filter.cluster_indices] += 1
            remain_filters[l_ind].append(filter.filter_index)

        remain_filters[l_ind].sort()

    
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)

        uncovered_cluster_indice.append(selector.remain_coverage_points)
    
    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)

    print('remain_filters', remain_filters)
    print('len(remain_filters', len(remain_filters))
    exit()
    
    # get cfg
    cfg = []
    max_pooling = [2,4,7,10]
    for n, m in enumerate(mid_cfg):
        if n in max_pooling:
            cfg.append("M")
            cfg.append(m)
        else:
            cfg.append(m)

    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)
    cprint("finish vgg pruning","r")

elif args.arch == "resnet56":
    cprint("start resnet56 pruning","r")
    extracted_filter_weight = []
    except_ = [0,21,40] 
    print("origin weight num: ", len(filter_weight_flattened_dict)) # 57
    for weight_num in weight_num_list:
        if weight_num in except_:
            pass
        else:
            extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
    print("extracted weight num: ", len(extracted_filter_weight)) # 57-3 = 54개


    ### ### ### ### 박윤아 작성 ### ### ### ### ### ### ### ### ### ### ### ###
    # extracted_filter_weight: 리스트의 한 element가 한 레이어에 대한 weight를 담고 있음.
    #  -> 리스트의 원소는 54개
    #  -> 한 element shape (filter 수, kernel 수 * 9)  # 여기에서 커널의 수 = 필터의 채널
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###



   
    # extract 3*3 conv
    mid_filter = []

    # 54번 반복문이 돌아간다. # mid_filter에 각 레이어 별 weight 담기
    for n in range(len(extracted_filter_weight)):
        # mid_filter에는 짝수번째 layer 그리고 0,21,40번째 레이어를 제외한 레이어의 weight가 들어간다. 
        if n%2 == 0:
            mid_filter.append(extracted_filter_weight[n])

    ### mid_filter 목록(숫자 + layer) / filter 26개 
    # /0 2 4 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 /40 42 44 46 48 50 52 54 56
    cprint("################# filter pruning- mid_filter추출완료 ##################", 'b')
    
    

    ### ### ### ### 박윤아 작성 ### ### ### ### ### ### ### ### ### ### ### ###
    # mid_filter: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 리스트의 원소는 27개 (짝수_layer 27개, 홀수_layer 27개)
    #  -> 한 element shape (filter 수, kernel 수 * 9) # kernel 수 = 필터의 채널 수 
    #  -> 해당 위치에 해당하는 layer의 weight 값이 존재. 
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


    ########################## calculate kerenl distance########################
    # mid 3*3 conv
    cprint("################# filter pruning- mid_distances 계산 시작 ##################", 'b')

    total_mid_distances = None

    print("calculate distance threshold/ 각 레이어 별로 distance들 모으기")
    for n in tqdm(range(len(mid_filter))): #27번 반복 

        # mid_distances: cluster 내의 ditance들을 원소로 가지는 배열
        mid_distances = cal_distance(mid_filter[n], 3) 

        #print('length',len(mid_distances)) #240
        if total_mid_distances is None:
            total_mid_distances = mid_distances
        else:
            total_mid_distances = np.concatenate((total_mid_distances, mid_distances), axis=0)

    # total_mid_distances: 모든 레이어에서의 cluster 내의 distance들을 원소로 가지는 배열
    thr = cdf(total_mid_distances,args.prune_ratio, args.threshold_type) #0.13840059809465113
    print("new thr : ", thr) #0.13840059809465113
    cprint("distance 계산 완료, threshold 추출완료", 'b')


    cprint("################# filter pruning- agglomerative clustering ##################", 'b')
    ######################### agglomerative clustring ############################
    # mid 3*3conv
    print("filter pruning")
    mid_cfg = []
    labels = []
    # 레이어의 개수만큼(27) 반복문이 돌아간다. # n은 0부터 26까지
    for n in tqdm(range(len(mid_filter))):

        filter_nums,label = newpruning(mid_filter[n], n, 3, thr, args.threshold_type) 
        ### ### ### ### newpruning의 output ### ### ### ### ### ### ### ### ### ###
        # filter_nums = max_filter_num: 주어진 n번째 레이어의 모든 커널 축을 고려했을때, 최대 cluster 개수
        # label = filter_label: 리스트의 원소는 주어진 n번째 레이어에서의 cluster 라벨들' 이다. 

        #print(' 첫번째 레이어_filter_nums', filter_nums)
        #print('첫번째 레이어_label', label)
        #print('첫번째 레이어_label.shape', label.shape) #(16, 16)

        labels.append(label) #labels : 리스트의 원소개수는, 레이어의 전체 수와 같고
        # mid_cfg: 리스트의 원소개수는, 레이어의 전체 수(27)와 같고, 각 원소는 채널 별 클러스터의 라벨 개수의 최대 값이다. 
        # '채널 별 클러스터의 라벨 개수' = '채널 별 클러스터의 개수' 의 최대 값이 어디에 활용되는건지 아직 모르겠음!
        mid_cfg.append(filter_nums) 


    print('mid_cfg', mid_cfg)
    layer_num = len(mid_filter)
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []

    #print('layer_num', layer_num) #27
    #print('labels', labels) # len(labels) = 27
 

    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        # coverage_points_num_kernels = [[], [], ..., [], []]
        layer_kernel = mid_filter[l_ind] # 한 레이어 안의 weight들을 담고 있음
        layer_label = labels[l_ind] 


        # t_layer_label : layer_label을 transpose한 것 
        t_layer_label = layer_label.T # np.T
        # (kernel 수, filter 수)= (b, a)
        kernel_num = t_layer_label.shape[0] # b
        print('t_layer_label', t_layer_label)

        accumulated_index = 0
        for k_col in range(kernel_num):

            # t_layer_label[k_col]: k번째 커널의 라벨, 형태는 리스트이다.
            # 리스트의 원소 개수는 필터의 개수와 같다. # 16, 32, 64
            print('t_layer_label[k_col]', t_layer_label[k_col])
                  
            t_layer_label[k_col] += accumulated_index
            # 리스트의 모든 원소에 연산을 한다. 

            print('accumulated_index', accumulated_index)
            print('t_layer_label[k_col]', t_layer_label[k_col])

            col_kernels = t_layer_label[k_col]

            # col_kernels_num:해당 레벨의 모든 커널축의 라벨들의 개수(중복 허용x)
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)
            # coverage_points_num_kernel_axis[l_ind] = [ a, b, c, ... ]
            # - > 'l_ind'번째 레이어에서 a번째 커널축까지의 cluster의 총 개수
            print('coverage_points_num_kernel_axis', coverage_points_num_kernel_axis)
            # coverage_points_num_kernel_axis 
            # [[0, 5, 6, 16, 22, 30, 32, 33, 41, 50, 61, 62, 66, 68, 69, 70], # 이 리스트는 해당 레이어의 각 커널축의 coverage point이다.  
            # [0, 5, 6, 15, 19, 29, 30, 32, 36, 40, 47, 50, 54, 57, 58, 60], 
            # [0, 7, 8, 15, 18, 24, 26, 28, 33, 39, 47, 51, 55, 60, 61, 65], 

            # [2, 5, 9, 12, 14, 15, 17, 20, 22, 24, 26, 28, 30, 31, 33, 35, 38, 39, 41, 44, 46, 49, 50, 53, 55, 57, 59, 62, 64, 66, 68, 69, 71, 73, 75, 78, 81, 84, 86, 87, 89, 91, 93, 95, 97, 99, 102, 104, 106, 108, 110, 112, 113, 115, 117, 119, 121, 122, 125, 129, 132, 134, 136, 138]]

        coverage_points_num.append(accumulated_index-1)
        print('coverage_points_num', coverage_points_num)
        # coverage_points_num 
        # [70, 60, 65, 87, 86, 90, 102, 111, 109, 184, 167, 244, 236, 166, 197, 161, 135, 119, 434, 540, 601, 560, 516, 410, 274, 455, 138]
        ## 리스트의 원소는 해당 레이어의 coverage point이다.  


        ### ### ### ### ### ### ### ### coverage point가 무엇인가 ? ### ### ### ###
        # t번째 커널축의 coverage point - t-1 번째 커널축의 coverage point = t번째 커널의 클러스터 개수  




    # Maximum coverage
    print("#################  filter pruning - Maximum Coverage ################")
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0

    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    remain_filters = []
    node_used_count = []
    uncovered_cluster_indice = []
    
    # 모든 레이어에 대해서 순차적으로 적용한다. 
    for l_ind in range(layer_num):
        # node_used_count: 리스트의 원소는 layer 별 클러스터의 개수만큼 0으로 채운 리스트형 배열
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        

        # remain_filters = [[]]
        remain_filters.append([])
        # 레이어의 개수만큼 []를 가지는 리스트 
       
        
        ### ### ### ### ### ### NaiveSelector의 return값을 아직 모르겠음 일단 아래를읽어보자 ### ### ### 
        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
        #print('첫번째 레이어의 selector.remain_coverage_points', selector.remain_coverage_points)
        
        # remain_filters[l_ind]= []
        # len(remain_filters[l_ind])=0
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 왜 이런 조건을 쓰는지 잘 모르겠음 
        # mid_cfg[l_ind]: 해당 레이어에서 채널 별 클러스터의 개수의 최댓값 
        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            #print('l_ind', l_ind)
            #print('len(remain_filters)', len(remain_filters))
            #print('mid_cfg[l_ind]', mid_cfg[l_ind])
            filter = selector.get_filter()
            #print('filter 출력')  
            #print('filter', filter) #filter <select_filter.FilterWithoutNorm object at 0x7f20852fc9d0>
            
            # return: False 


            # filter.cluster_indices = selector.get_filter.cluster_indices
            # cluster_indices : 해당 레이어에서 클러스터의 라벨들 

            # node_used_count[l_ind][filter.cluster_index]: 해당 레이어의 해당하는 클러스터 라벨에 해당하는 데이터의 수 
            node_used_count[l_ind][filter.cluster_indices] += 1

            # filter.filter_index: 
            remain_filters[l_ind].append(filter.filter_index)
            #print('while문 안에서', remain_filters)
        
        remain_filters[l_ind].sort()

       
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)
        

        #print('마지막 레이어의 selector.remain_coverage_points', selector.remain_coverage_points) 
        # {54}

        uncovered_cluster_indice.append(selector.remain_coverage_points)
        #print('uncovered_cluster_indice', uncovered_cluster_indice)
        

    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)
        
    # get cfg
    cfg = [16]
    for n, m in enumerate(mid_cfg):
        if n < 9:
            cfg.append(m)
            cfg.append(16)
        elif n >=9 and n <18:
            cfg.append(m)
            cfg.append(32)
        elif n >=18 and n <27:
            cfg.append(m)
            cfg.append(64)
            
    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)
    # [16, 11, 16, 10, 16, 8, 16, 11, 16, 12, 16, 12, 16, 12, 16, 15, 16, 15, 16, 21, 32, 9, 32, 14, 32, 14, 32, 11, 32, 14, 32, 8, 32, 8, 32, 7, 32, 20, 64, 14, 64, 15, 64, 13, 64, 12, 64, 10, 64, 7, 64, 10, 64, 4, 64]
    cprint("finish resnet56 pruning","r")

elif args.arch == "resnet110":
    # extract downsample conv & first conv
    cprint("start resnet110 pruning","r")
    extracted_filter_weight = []

    except_ = [0,39,76]
    print("origin weight num: ", len(filter_weight_flattened_dict))
    for weight_num in weight_num_list:
        if weight_num in except_:
            pass
        else:
            extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
    print("extracted weight num: ", len(extracted_filter_weight))

    # extract 3*3 conv
    mid_filter = []
    for n in range(len(extracted_filter_weight)):
        if n%2 == 0:
            mid_filter.append(extracted_filter_weight[n])
    print("layer num: ", len(mid_filter))

    # calculate kerenl distance
    # mid 3*3 conv 

    total_mid_distances = None

    print("calculate distance threshold")
    for n in tqdm(range(len(mid_filter))):
        mid_distances = cal_distance(mid_filter[n], 3)
        if total_mid_distances is None:
            total_mid_distances = mid_distances
        else:
            total_mid_distances = np.concatenate((total_mid_distances, mid_distances), axis=0)

    thr = cdf(total_mid_distances,args.prune_ratio, args.threshold_type)
    print("new thr : ", thr)

    # agglomerative clustring
    # mid 3*3conv

    print("filter pruning")
    mid_cfg = []
    labels = []
    for n in tqdm(range(len(mid_filter))):
        filter_nums,label = newpruning(mid_filter[n], n, 3, thr, args.threshold_type) 

        labels.append(label)
        mid_cfg.append(filter_nums)

    layer_num = len(mid_filter)
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []


    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        layer_kernel = mid_filter[l_ind]
        layer_label = labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)

        #print(accumulated_index)
        coverage_points_num.append(accumulated_index-1)

    # Maximum coverage
    print("Maximum Coverage")
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    remain_filters = []
    node_used_count = []
    uncovered_cluster_indice = []
    for l_ind in range(layer_num):
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        remain_filters.append([])

        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)


        # mid_cfg: 리스트의 원소개수는, 레이어의 전체 수(27)와 같고, 각 원소는 채널 별 클러스터의 라벨 개수의 최대 값이다.
        # len(remain_filters[l_ind] = l번째 레이어에서 'l'을 의미한다. )
        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):

            filter = selector.get_filter()
            # return max filter

            node_used_count[l_ind][filter.cluster_indices] += 1

            remain_filters[l_ind].append(filter.filter_index)
        remain_filters[l_ind].sort()
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)

        uncovered_cluster_indice.append(selector.remain_coverage_points)
    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)

    # get cfg
    cfg = [16]
    for n, m in enumerate(mid_cfg):
        if n < 18:
            cfg.append(m)
            cfg.append(16)
        elif n >=18 and n <36:
            cfg.append(m)
            cfg.append(32)
        elif n >=36 and n <54:
            cfg.append(m)
            cfg.append(64)
            
    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)
    cprint("finish resnet110 pruning","r")

elif args.arch == "resnet18":
    # resnet18
    extracted_filter_weight = []
    except_ = [0,7,12,17]
    print("origin weight num: ", len(filter_weight_flattened_dict))
    for weight_num in weight_num_list:
        if weight_num in except_:
            pass
        else:
            extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
    print("extracted weight num: ", len(extracted_filter_weight))

    # extract 3*3 conv
    mid_filter = []
    for n in range(len(extracted_filter_weight)):
        if n%2 == 0:
            mid_filter.append(extracted_filter_weight[n])
    print("layer num: ", len(mid_filter))

    # calculate kerenl distance
    # mid 3*3 conv

    print("calculate distance threshold")
    total_mid_distances = []
    for n in tqdm(range(len(mid_filter))):
        mid_distances = cal_distance(mid_filter[n], 3)
        total_mid_distances.extend(mid_distances)

    thr = cdf(total_mid_distances,args.prune_ratio)
    print("new thr : ", thr)

    # agglomerative clustring
    # mid 3*3conv
    print("filter pruning")
    mid_cfg = []
    labels = []
    for n in tqdm(range(len(mid_filter))):
        filter_nums,label = newpruning(mid_filter[n], n, 3, thr)

        labels.append(label)
        mid_cfg.append(filter_nums)
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # mid_filter: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수 * 9)
    #  -> 해당 위치에 해당하는 kernel의 weight 값이 존재.  
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # labels: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수)
    #  -> 해당 위치에 해당하는 kernel이 어느 cluster에 속하는지 cluster index 정보가 존재. 
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

    layer_num = len(mid_filter)
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []


    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        layer_kernel = mid_filter[l_ind]
        layer_label = labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)


        #print(accumulated_index)
        coverage_points_num.append(accumulated_index-1)

    # Maximum coverage
    print("Maximum Coverage")
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    remain_filters = []
    node_used_count = []
    uncovered_cluster_indice = []
    for l_ind in range(layer_num):
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        remain_filters.append([])

        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)

        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            # args.criterion = 'reprune' # get_filter()의 출력:self._get_max_coverage_filter()
            filter = selector.get_filter()

            node_used_count[l_ind][filter.cluster_indices] += 1

            remain_filters[l_ind].append(filter.filter_index)

        remain_filters[l_ind].sort()
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)

        uncovered_cluster_indice.append(selector.remain_coverage_points)
    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)

    cfg = [64]
    print("Mid cfg len : ", len(mid_cfg))
    for n, m in enumerate(mid_cfg):
        if n < 2:
            cfg.append(m)
            cfg.append(64)
        elif n >=2 and n <4:
            cfg.append(m)
            cfg.append(128)
        elif n >=4 and n <6:
            cfg.append(m)
            cfg.append(256)
        elif n >=6 and n <8:
            cfg.append(m)
            cfg.append(512)


    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)

elif args.arch == "resnet50":
    cprint("start resnet50 pruning","r")
    extracted_filter_weight = []

    except_ = [0,4,14,27,46]
#    end = 42
    print("origin weight num: ", len(filter_weight_flattened_dict))
    for weight_num in weight_num_list:
        if weight_num in except_:
            pass
        else:
            extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
    print("extracted weight num: ", len(extracted_filter_weight))

    # extract front 1*1 conv
    bottleneck_filter = []
    bottleneck_channels_pruned_layer=[]

    for n in range(len(extracted_filter_weight)):
        if n%3 == 0:
            bottleneck_channels_pruned_layer.append(extracted_filter_weight[n].shape[0])
            bottleneck_filter.append(extracted_filter_weight[n])

    # extract 3*3 conv

    mid_filter = []
    mid_channels_pruned_layer=[]
    for n in range(len(extracted_filter_weight)):
        if n%3 == 1:
            mid_channels_pruned_layer.append(extracted_filter_weight[n].shape[0])
            mid_filter.append(extracted_filter_weight[n])

    total_bottleneck_distances = []
    for n in tqdm(range(len(bottleneck_filter))):
        bottleneck_distances = cal_distance(bottleneck_filter[n], 1)
        total_bottleneck_distances.extend(bottleneck_distances)

    bottleneck_thr = cdf(total_bottleneck_distances,args.prune_ratio)
    print("new bottleneck thr : ", bottleneck_thr)

    # mid 3*3 conv 
    total_mid_distances = []
    for n in tqdm(range(len(mid_filter))):
        mid_distances = cal_distance(mid_filter[n], 3)
        total_mid_distances.extend(mid_distances)

    mid_thr = cdf(total_mid_distances,args.prune_ratio)
    print("new mid thr : ", mid_thr)

    # agglomerative clustring
    # front 1*1 conv
    mid_labels = []
    bottleneck_labels = []
    bottleneck_cfg = []
    mid_cfg = []

    for n in tqdm(range(len(mid_filter))):
        mid_filter_nums,mid_label = newpruning(mid_filter[n], n, 3, mid_thr)
        bottleneck_filter_nums, bottleneck_label = detect_bottleneck_label(bottleneck_filter[n], 1, mid_filter_nums)
        mid_cfg.append(mid_filter_nums)
        bottleneck_cfg.append(bottleneck_filter_nums)

        mid_labels.append(mid_label)
        bottleneck_labels.append(bottleneck_label)

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # mid_filter: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수 * 9)
    #  -> 해당 위치에 해당하는 kernel의 weight 값이 존재.  
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # labels: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수)
    #  -> 해당 위치에 해당하는 kernel이 어느 cluster에 속하는지 cluster index 정보가 존재. 
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

    layer_num = len(mid_filter)
    remain_filters = []
    bottleneck_remain_filters = []
    coverage_points_num = []
    bottleneck_coverage_points_num = []


    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여 (1x1)
    for l_ind in range(layer_num):

        layer_kernel = bottleneck_filter[l_ind]
        layer_label = bottleneck_labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num

        #print(accumulated_index)
        bottleneck_coverage_points_num.append(accumulated_index-1)
    

    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여 (3x3)
    for l_ind in range(layer_num):

        layer_kernel = mid_filter[l_ind]
        layer_label = mid_labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num

        #print(accumulated_index)
        coverage_points_num.append(accumulated_index-1)
    
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0
    bottleneck_total_nodes_to_cover = 0
    bottleneck_total_uncovered_nodes = 0
    # Maximum coverage
    print("Maximum Coverage")
    # 해당 레이어, 필터에서의 maximum coverage (1x1)
    print("1x1")
    print("bottleneck config: ", bottleneck_cfg)
    from select_filter import NaiveSelector
    for l_ind in range(layer_num):
        bottleneck_remain_filters.append([])

        selector = NaiveSelector(bottleneck_labels[l_ind], bottleneck_filter[l_ind], bottleneck_coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
        print("Layer ", l_ind, "Cover size: ", selector.total_nodes_num)
        while(len(bottleneck_remain_filters[l_ind]) < bottleneck_cfg[l_ind]):
            filter = selector.get_filter()

            bottleneck_remain_filters[l_ind].append(filter.filter_index)
        
        bottleneck_remain_filters[l_ind].sort()
        bottleneck_total_nodes_to_cover += selector.total_nodes_num
        bottleneck_total_uncovered_nodes += len(selector.remain_coverage_points)
    print("1x1 Total uncovered nodes ratio: ", bottleneck_total_uncovered_nodes / bottleneck_total_nodes_to_cover * 100)

    # 해당 레이어, 필터에서의 maximum coverage(3x3)
    print("3x3")
    for l_ind in range(layer_num):
        remain_filters.append([])
        #print("The number of points to cover", max(labels[l_ind].flatten()))

        selector = NaiveSelector(mid_labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
        #print("Layer ", l_ind, "Cover size: ", selector.total_nodes_num)

        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            filter = selector.get_max_coverage_filter()

            remain_filters[l_ind].append(filter.filter_index)
        
        # print(len(remain_filters[l_ind]), len(selector.remain_coverage_points))
        remain_filters[l_ind].sort()
        #print("Uncovered nodes ratio: ")
        #print(len(selector.remain_coverage_points) / selector.total_nodes_num * 100)
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)
    print("3x3 Total uncovered nodes ratio: ", total_uncovered_nodes / total_nodes_to_cover * 100)
    
    # get cfg
    cfg = [64]
    for b,m in zip(bottleneck_cfg,mid_cfg):
        cfg.append(b)
        cfg.append(m)

    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)

elif args.arch == "densenet40":
    cprint("start densenet pruning","r")
    extracted_filter_weight = []
    except_ = [0,13,26] # !
    print("origin weight num: ", len(filter_weight_flattened_dict))
    for weight_num in weight_num_list:
        if weight_num in except_:
            pass
        else:
            extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
    print("extracted weight num: ", len(extracted_filter_weight))

    # extract 3*3 conv
    mid_filter = []
    for n in range(len(extracted_filter_weight)):
        mid_filter.append(extracted_filter_weight[n])
    print("layer num: ", len(mid_filter))

    # calculate kerenl distance
    # mid 3*3 conv 
    total_mid_distances = None

    print("calculate distance threshold")
    for n in tqdm(range(len(mid_filter))):
        mid_distances = cal_distance(mid_filter[n], 3)                       
        if total_mid_distances is None:
            total_mid_distances = mid_distances
        else:
            total_mid_distances = np.concatenate((total_mid_distances, mid_distances), axis=0)


    thr = cdf(total_mid_distances, args.prune_ratio, args.threshold_type)
    print("new thr : ", thr)
    # agglomerative clustring
    # mid 3*3conv
    

    print("filter pruning")
    mid_cfg = []
    labels = []
    for n in tqdm(range(len(mid_filter))):
        filter_nums,label = newpruning(mid_filter[n], n, 3, thr, args.threshold_type)    
        mid_cfg.append(filter_nums)
        labels.append(label)

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # mid_filter: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수 * 9)
    #  -> 해당 위치에 해당하는 kernel의 weight 값이 존재.  
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    # labels: 리스트의 한 element가 레이어에 대한 정보를 담고 있음.
    #  -> 한 element shape (filter 수, kernel 수)
    #  -> 해당 위치에 해당하는 kernel이 어느 cluster에 속하는지 cluster index 정보가 존재. 
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

    layer_num = len(mid_filter)
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []

    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        layer_kernel = mid_filter[l_ind]
        layer_label = labels[l_ind]

        t_layer_label = layer_label.T
        # (kernel 수, filter 수): kernel이 속한 cluster index
        kernel_num = t_layer_label.shape[0]

        accumulated_index = 0
        for k_col in range(kernel_num):
            t_layer_label[k_col] += accumulated_index
            col_kernels = t_layer_label[k_col]
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)

        #print(accumulated_index)
        coverage_points_num.append(accumulated_index-1)

    # Maximum coverage
    print("Maximum Coverage")
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    # Code book (cluster_num, mid_filters, labels)
    codebook_list = []
    remain_filters = []
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0
    node_used_count = []
    uncovered_cluster_indice = []

    # Maximum coverage
    print("Maximum Coverage")
    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector

    for l_ind in range(layer_num):
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        remain_filters.append([])
        
        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            max_filter = selector.get_filter()

            node_used_count[l_ind][max_filter.cluster_indices] += 1
            remain_filters[l_ind].append(max_filter.filter_index)
        remain_filters[l_ind].sort()

        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)

        uncovered_cluster_indice.append(selector.remain_coverage_points)
    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)

    # get cfg
    cfg = []
    for n, m in enumerate(mid_cfg):
        cfg.append(m)

    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)
    cprint("finish vgg pruning","r")

elif args.arch == "mobilenetv2":
    cprint("start mobilenetv2 pruning","r")
    extracted_filter_weight =  []
    weight_layer = [i for i in range(1,50,3)] 
    print(weight_layer)

    for weight_num in weight_layer:
        extracted_filter_weight.append(filter_weight_flattened_dict[weight_num])
        print('shape 출력', filter_weight_flattened_dict[weight_num].shape)
    print("extracted weight num: ", len(extracted_filter_weight)) # 17개

    # extract 3*3 conv
    mid_filter = []

    for n in range(len(extracted_filter_weight)):
            mid_filter.append(extracted_filter_weight[n])

    print("layer num: ", len(mid_filter)) #17

    cprint("################# filter pruning- mid_filter추출완료 ##################", 'b')


    ########################## calculate kerenl distance########################
    # mid 3*3 conv
    cprint("################# filter pruning- mid_distances 계산 시작 ##################", 'b')

    total_mid_distances = None



    print("calculate distance threshold/ 각 레이어 별로 distance들 모으기")
    for n in tqdm(range(len(mid_filter))): 

        # mid_distances: cluster 내의 ditance들을 원소로 가지는 배열 # input: 레이어, 커널사이즈
        mid_distances = cal_distance(mid_filter[n], 1) 

        #print('length',len(mid_distances)) #240
        if total_mid_distances is None:
            total_mid_distances = mid_distances
        else:
            total_mid_distances = np.concatenate((total_mid_distances, mid_distances), axis=0)

    # total_mid_distances: 모든 레이어에서의 cluster 내의 distance들을 원소로 가지는 배열
    thr = cdf(total_mid_distances, args.prune_ratio, args.threshold_type) #0.13840059809465113
    print("new thr : ", thr) #0.13840059809465113
    cprint("distance 계산 완료, threshold 추출완료", 'b')


    cprint("################# filter pruning- agglomerative clustering ##################", 'b')
    # mid 3*3conv
    print("filter pruning")
    mid_cfg = []
    labels = []
    # 레이어의 개수만큼(17) 반복문이 돌아간다.
    for n in tqdm(range(len(mid_filter))):

        ### ### ###  커널 크기는 1
        filter_nums,label = newpruning(mid_filter[n], n, 1, thr, args.threshold_type) 
        ### ### ### ### newpruning의 output ### ### ### ### ### ### ### ### ### ###
        # filter_nums = max_filter_num: 주어진 n번째 레이어의 모든 커널 축을 고려했을때, 최대 cluster 개수 # 하나의 정수
        print('label.shape', label.shape)
        #label.shape #(필터의 개수, 필터의 채널 수) # 이때 필터의 '채널 수'는 전 레이어로부터 정해지는 값이다. 

        labels.append(label) #labels : 리스트의 원소개수는, 레이어의 전체 수와 같고
        # mid_cfg: 리스트의 원소개수는, 레이어의 전체 수(17)와 같고, 각 원소는 레이어 별 클러스터의 라벨 개수의 최대 값이다. 
        # '채널 별 클러스터의 라벨 개수' = '채널 별 클러스터의 개수' 의 최대 값이 어디에 활용되는건지 아직 모르겠음!
        mid_cfg.append(filter_nums)  #[30, 93, 544, 144, 189, 191, 192, 549, 352, 348, 367, 166, 82, 556, 27, 22, 930]

    #print('mid_cfg', mid_cfg)
    
    
    layer_num = len(mid_filter) # 17
    remain_filters = []
    coverage_points_num = []
    coverage_points_num_kernel_axis = []

 

    # 커널 축 별 독립된 cluster에 연속된 cluster index 부여
    for l_ind in range(layer_num):
        coverage_points_num_kernel_axis.append([])
        # coverage_points_num_kernels = [[], [], ..., [], []]

        layer_kernel = mid_filter[l_ind] # 한 레이어 안의 weight들을 담고 있음
        #print('layer_kernel',layer_kernel,'layer.shape',  layer_kernel.shape )

        layer_label = labels[l_ind] 
        # layer_label.shape : (필터의 input 채널 수, 필터의 개수(output 채널 수)) # (96, 16)
        #print('layer_label',layer_label,'label.shape', layer_label.shape )
        
        # t_layer_label : layer_label을 transpose한 것 
        t_layer_label = layer_label.T # np.T # filter의 원래 차원으로 다시 transpose (filter의 채널 수, filter의 개수)
        #print('t_layer.label', t_layer_label)

        # t_layer_label.shape :(필터의 개수, 필터의 input 채널 수)
        kernel_num = t_layer_label.shape[0] # b= 필터의 input 채널 수 = kernel축의 개수
        print('t_layer_label', t_layer_label.shape)

        accumulated_index = 0
        for k_col in range(kernel_num):
            #print('{} layer and {} kernel'.format(l_ind, k_col))
                  
            t_layer_label[k_col] += accumulated_index
            # 리스트의 모든 원소에 연산을 한다. 

            #print('accumulated_index', accumulated_index)
           
            col_kernels = t_layer_label[k_col]

            # col_kernels_num:해당 커널축의 클러스터 개수
            col_kernels_num = len(set(col_kernels))

            accumulated_index += col_kernels_num
            coverage_points_num_kernel_axis[l_ind].append(accumulated_index-1)
            # coverage_points_num_kernel_axis[l_ind] = [ 1, 2,, n... ]
            # 원소 n- > 'l_ind'번째 레이어에서 1번째 부터 n번째 커널축까지의 cluster의 총 개수
            #print('{}th layer and {}th kernel'.format(l_ind, k_col))
            #print('coverage_points_num_kernel_axis', coverage_points_num_kernel_axis)

            # coverage_points_num_kernel_axis 
            #[[28, 58, 87, 115], [91, 184], 
            # [542, 1076, 1617, 2148, 2689, 3233, 3763, 4302, 4845, 5385, 5906], [143, 287, 431], 
            # [187, 376, 565, 745], [188, 374, 565, 731], 
            # [191, 382, 574, 756], [548, 1082, 1613, 2150, 2687, 3222, 3756, 4285, 4826, 5360, 5863], 
            # [346, 698, 1040, 1388, 1737, 2076, 2416, 2477], [343, 691, 1029, 1369, 1716, 2053, 2393, 2449], 
            # [360, 725, 1079, 1441, 1808, 2163, 2520, 2607], [146, 299, 458, 604, 746, 882, 1048, 1189, 1339, 1490, 1580], 
            # [69, 142, 215, 289, 358, 429, 511, 585, 655, 724, 773], 
            # [548, 1098, 1652, 2207, 2757, 3305, 3861, 4416, 4968, 5520, 6047], 
            # [24, 49, 73, 100, 123, 143, 164, 186, 209, 231, 254, 275, 296, 320, 341, 365, 390, 407], 
            # [20, 37, 58, 78, 98, 114, 131, 146, 164, 180, 199, 217, 235, 249, 268, 286, 308, 321], 
            # [920, 1843, 2763, 3693, 4615, 5532, 6445, 7365, 8284, 9208, 10121, 11039, 11960, 12881, 13805, 14725, 15652, 16564]] 


        coverage_points_num.append(accumulated_index-1)
    
    print('coverage_points_num', coverage_points_num) # 
    # coverage_points_num 
    # [115, 184, 5906, 431, 745, 731, 756, 5863, 2477, 2449, 2607, 1580, 773, 6047, 407, 321, 16564]
    ### coverage_points_num 의 원소
    # t번째 레이어의 coverage point의 개수 - t-1 번째 레이어의 coverage point의 개수 = t번째 커널의 클러스터 개수  




    # Maximum coverage
    print("#################  filter pruning - Maximum Coverage ################")
    total_nodes_to_cover = 0
    total_uncovered_nodes = 0


    # 해당 레이어, 필터에서의 maximum coverage
    from select_filter import NaiveSelector
    remain_filters = []
    node_used_count = []
    uncovered_cluster_indice = []
    
    # 모든 레이어에 대해서 순차적으로 적용한다. 
    for l_ind in range(layer_num):
        # node_used_count: 리스트의 원소는 layer 별 '클러스터의 개수 + 1' 만큼 0으로 채운 리스트형 배열
        node_used_count.append(np.array([0 for i in range(coverage_points_num[l_ind]+1)]))
        
        remain_filters.append([])
        # 레이어의 개수(17개)만큼 []를 가지는 리스트 
       
        
        ### ### ### ### ### ### NaiveSelector의 return값을 아직 모르겠음 일단 아래를읽어보자 ### ### ### 
        selector = NaiveSelector(labels[l_ind], mid_filter[l_ind], coverage_points_num[l_ind], args.criterion, args.selection_with_norm)
       
        
        # remain_filters[l_ind]= []
        # len(remain_filters[l_ind])=0
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 왜 이런 조건을 쓰는지 잘 모르겠음 
        # mid_cfg[l_ind]: 해당 레이어에서 채널 별 클러스터의 개수의 최댓값 
        while(len(remain_filters[l_ind]) < mid_cfg[l_ind]):
            filter = selector.get_filter()
    
            # cluster_indices : 해당 레이어에서 클러스터의 라벨들 
            # node_used_count[l_ind][filter.cluster_index]: 해당 레이어의 해당하는 클러스터 라벨에 해당하는 데이터의 수 
            node_used_count[l_ind][filter.cluster_indices] += 1

            # filter.filter_index: 
            remain_filters[l_ind].append(filter.filter_index)
            #print('while문 안에서', remain_filters)
        
        remain_filters[l_ind].sort()

       
        total_nodes_to_cover += selector.total_nodes_num
        total_uncovered_nodes += len(selector.remain_coverage_points)
        

        uncovered_cluster_indice.append(selector.remain_coverage_points)
        #print('uncovered_cluster_indice', uncovered_cluster_indice)
        

    uncovered_ratio = total_uncovered_nodes / total_nodes_to_cover * 100
    print("Total uncovered nodes ratio: ", uncovered_ratio)
        
    # get cfg
    cfg = [32]
    channel = [16, 24, 24, 32, 32, 32, 96, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
    for i in range(layer_num): # layer_num: 17
        cfg.append(mid_cfg[i])
        cfg.append(mid_cfg[i])
        cfg.append(channel[i])


    original_cfg =[[32],[32, 32, 16], [96, 96, 24], [144, 144, 24], [144, 144, 32], [192, 192, 32], [192, 192, 32],
                 [192, 192, 64], [384, 384, 64], [384, 384, 64], [384, 384, 64], [384, 384, 96], [576, 576, 96],
                 [576, 576, 96], [576, 576, 160], [960, 960, 160], [960, 960, 160], [960, 960 ,320], [1280]]

            
    print("final cfg num: ", len(cfg))
    print("final cfg : ", cfg)
    cprint("finish resnet56 pruning","r")
    # [32, 32, 32, 16, 96, 96, 24, 143, 143, 24, 144, 144, 32, 190, 190, 32, 188, 188, 32, 
    # 190, 190, 96, 364, 364, 64, 349, 349, 64, 342, 342, 64, 360, 360, 96, 404, 404, 96,
    # 361, 361, 96, 518, 518, 160, 170, 170, 160, 123, 123, 160, 783, 783, 320]


    for_build_remain_filters = remain_filters
    for_build_labels = labels




################################# filter pruning ##########################################

################################# load model ##########################################
# init
cprint("load model","r")
if args.dataset == 'imagenet':
    # architecture가 'resnet'이면
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](cfg = cfg)

    else:
        model = models.__dict__[args.arch](batch_norm=True, cfg = cfg)
    
    # torch.randn: normal random 분포로부터 랜덤으로, 주어진 사이즈를 갖는 텐서 출력
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 256, 256]))

elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
    kwargs = {'dataset': args.dataset, 'cfg' : cfg}
    model = models.__dict__["MobileNetV2"](**kwargs)
    
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]))
    #print('flops', flops) #99869312
    #print('params', params) #575034
    #print('results', results) # 노션


################################# load model ##########################################

################################# load weight ##########################################

## load pretrained weights
cprint("load weight","r")

if args.finetuning:
    print(model)
    print('build 입장')
    # 모델의 파라미터 불러오기 # 'state_dice' 모델 구조 전체가 아닌 파라미터를 불러온다. 
    # args.pretrained : "./logs/mobilenetv2/{}_baseline_{}/model_best.pth.tar".format(model_name,args.dataset)
    pretrained = torch.load(args.pretrained, map_location='cpu' if args.cuda else 'cpu')['state_dict']
    # arch, pretrained_weight, model, remain_filter_indices
    if 'resnet50' == args.arch:
        resnet50_remain_filters = []
        for i in range(len(remain_filters)):
            resnet50_remain_filters.append(bottleneck_remain_filters[i])
            resnet50_remain_filters.append(remain_filters[i])

        builder = Builder(arch=args.arch, pretrained_weight=pretrained, model=model, remain_filter_indices=resnet50_remain_filters, cuda=args.cuda, labels=labels)
    

    else: 
        builder = Builder(arch=args.arch, pretrained_weight=pretrained, model=model, remain_filter_indices=remain_filters, cuda=args.cuda, labels=labels)

    builder.bulid_model()
    print('build 완료')
    exit()

elif args.evaluate:
    print(model)
    try:
        savepath = os.path.join(args.save, "finetuning{}".format(args.seed), "{}".format(args.prune_ratio), 'model_best.pth.tar')
    except Exception as e:
        print(e)
        print("Weight not exists")
        exit()

    pretrained = torch.load(savepath, map_location='cpu' if args.cuda else 'cpu')['state_dict']
    model.load_state_dict(pretrained)

model = model.to(device)

if args.evaluate:
    if args.dataset == 'imagenet':
        flops, params, results = count_flops_params(model, torch.randn([128, 3, 256, 256]))

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]))

    prec1, prec5, test_loss = test(model)
    print("Top1: ", prec1, "Top5: ", prec5)
    print("flops: ", flops, "params: ", params)

    exit()
   
################################# load weight ##########################################

################################# train model ##########################################
cprint("start training","r")
if args.finetuning:
    savepath = os.path.join(args.save, "finetuning{}".format(args.seed),str(args.prune_ratio))
else:
    savepath = os.path.join(args.save, "init{}".format(args.seed),str(args.prune_ratio))

if os.path.isdir(savepath):
    print("weight already exists")
    exit()
else:
    os.makedirs(savepath)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
# optimizer = SGDP(model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas= (0.9,0.999))

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1)
criterion = nn.CrossEntropyLoss().to(device)

# debugin2

results = {}
results['Top1_TeAcc'] = list()
results['Top5_TeAcc'] = list()
results['TrLoss'] = list()
results['TeLoss'] = list()
best_prec1 = 0.
for i in range(args.epochs):
    train_loss = train(model,i ,optimizer, criterion)
    prec1, prec5, test_loss = test(model)

    results['Top1_TeAcc'].append(prec1)
    results['Top5_TeAcc'].append(prec5)
    results['TrLoss'].append(train_loss)
    results['TeLoss'].append(test_loss)

    pd.DataFrame(results).to_csv(os.path.join(savepath, "results.csv"), index=None)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': g_epoch + 1,
        'state_dict': model.state_dict(),
        'flops' : flops,
        'params' : params,
        'uncovered_ratio' : uncovered_ratio,
        'cfg' : cfg,
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=savepath)
    print("Best accuracy: "+str(best_prec1))
#    scheduler.step()
print("Finished saving training history")
################################# train model ##########################################
