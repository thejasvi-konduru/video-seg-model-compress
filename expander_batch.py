import os
import itertools
import json
import argparse
import subprocess

import numpy as np
import math

import utils
import torch

def dump_config_file(dataset, arch, pruner_type, oblock_size, cblock_size, iblock_size, osp, opat, isp, ipat,
                is_repetitive, collapse_tensor, cross_prob, is_symmetric, pconfig_path, input_config):

    # Getting the model
    model = utils.create_model(dataset, arch)
    # print(list(model.parameters())) //prints the weights of the layers
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print()
    # exit()

    rbgp_layers = []
    if arch in ["resnext50_32x4d", "resnet18"]:
        # Do not prune, first and last layer
        non_rbgp_layers = ["conv1","fc"]
    elif arch == "mobilenet_v2":
        # Do not prune, first and last layer
        non_rbgp_layers = ["features.0.0","classifier.1"]
    elif arch in ["cifar_res_rvgg11", "cifar_rvgg11"]:
        # Do not prune, first conv and classifier
        non_rbgp_layers = ["features.0", "classifier.6"]
    elif arch in ["cifar_vgg19_bn"]:
        non_rbgp_layers = ["features.0", "classifier.3"]
    elif arch in ["cifar_resnet20", "cifar_resnet18"]:
        non_rbgp_layers = ["conv1","fc"]
    elif arch in ["cifar_wrn_16_4", "cifar_wrn_16_10", "cifar_wrn_40_4", "cifar_wrn_28_10"]:
        non_rbgp_layers = ["conv1","linear"]
    elif arch in ["drn_d_22", "drn_d_54"]:
        non_rbgp_layers = ["layer0.0","fc"]


    # Check validity of non_rbgp_layers
    lnames = []
    for name,module in model.named_modules():
        lnames.append(name)
        print("name {}".format(name))
    assert([_ in lnames for _ in non_rbgp_layers])

    # Selecting layers to do rbgp
    rbgp_layers = []
    for name,module in model.named_modules():
        if type(module) == torch.nn.Linear or (type(module) == torch.nn.Conv2d and module.groups == 1):
            if name not in non_rbgp_layers:
                rbgp_layers.append("layer."+name.lstrip("layer")+".weight")
        elif type(module) in [torch.nn.Linear, torch.nn.Conv2d]:
            non_rbgp_layers.append(name)

    verbose = True
    if verbose:
        print("RBGP layers", rbgp_layers)
        print("Non RBGP layers", non_rbgp_layers)

    import collections
    ls_config = collections.OrderedDict()

#----------------------------------------------------------------------------#
    print("---------------------------------------")
    with open(input_config, 'r') as data_file:
        data_temp = json.load(data_file)

    for element in data_temp["configs"]:
        element.pop('make_kwargs')
        element.pop('exec_args')

    with open("temp"+input_config, 'w') as data_file:
        data_temp = json.dump(data_temp, data_file)

    print("---------------------------------------")  
#-----------------------------------------------------------------------------#
    # ls_config=data_temp["configs"]
    with open("temp"+input_config, 'r') as data_file:
        data_temp = json.load(data_file)
    temp=data_temp["configs"]
    # print(temp["layer_set"])
    # print(data_temp["configs"])
    # ls_config["obh"] = oblock_size[0]
    # ls_config["obw"] = oblock_size[1]
    # ls_config["cbh"] = cblock_size[0]
    # ls_config["cbw"] = cblock_size[1]
    # ls_config["ibh"] = iblock_size[0]
    # ls_config["ibw"] = iblock_size[1]
    # ls_config["osp"] = osp
    # ls_config["opat"] = opat
    # ls_config["isp"] = isp
    # ls_config["ipat"] = ipat
    # ls_config["is_repetitive"] = is_repetitive
    # ls_config["collapse_tensor"] = collapse_tensor

    # ls_config["cross_prob"] = cross_prob
    # ls_config["is_symmetric"] = is_symmetric

    # ls_config["layer_set"] = rbgp_layers

    # Constructing json
    data = collections.OrderedDict()
    data["pruner_type"] = "srmbrep"
    # data["configs"] = [ls_config]
    data["configs"]=temp

    ####### Specially handling 4x4 for sparsites > 87.5 #######
    if arch == "cifar_wrn_40_4" and \
        isp >= 0.875 and \
        iblock_size[0] == 4 and iblock_size[1] == 4:

        # Clone ls_config
        import copy

        rbgp_layers_2x2 = ["module.layer1.0.conv1.weight",
                        "module.layer1.0.shortcut.0.weight"]

        # Preparing 4x4
        ls_config_4x4 = copy.deepcopy(ls_config)
        rbgp_layers_4x4 = []
        for layer in ls_config["layer_set"]:
            if not layer in rbgp_layers_2x2:
                rbgp_layers_4x4.append(layer)
        ls_config_4x4["layer_set"] = rbgp_layers_4x4

        # Preparing 2x2
        ls_config_2x2 = copy.deepcopy(ls_config)
        ls_config_2x2["layer_set"] = rbgp_layers_2x2
        if isp == 0.875:
            ls_config_2x2["ibh"] = 2
            ls_config_2x2["ibw"] = 2
        else:
            ls_config_2x2["ibh"] = 1
            ls_config_2x2["ibw"] = 1


        data["configs"] = [ls_config_4x4, ls_config_2x2]


    #########################################################


    # Writing configuration file
    fh = open(pconfig_path,"w+")
    json.dump(data, fh, indent=4)
    fh.close()


def extract_accuracy(exp_dir):
    import os
    import torch
    return torch.load(os.path.join(exp_dir,"model_best.pth.tar"))["best_acc1"].item()


from tools.calculate_spectral_gap import extract_spectral_gap
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input_config", default = '', required = True)
    args = parser.parse_args()

    # Constants and information
    dataset_path_map = {"imagenet":"~/S2021/Imagenet2012",
                        "cifar10":"./data",
                        "cifar100":"./data",
                        "cityscapes":"/ssd_scratch/cvit/furqan.shaik/cityscapes/cityscapes"}

    input_config = args.input_config
    input_size = input_config.split("_")[3]
    osp_size = float(input_config.split("_")[4])
    isp_size = float(input_config.split("_")[5].split(".json")[0])
    print("Isp size",isp_size)
    # RBGP fixed configuration
    pruner_type = "srmbrep"
    opat = "RAMANUJAN"
    ipat = "RAMANUJAN"
    collapse_tensor = True  #Flatten
    is_repetitive = True
    cross_prob = 0.5
    is_symmetric = False

    # Type of experiments (Training, Finetuning) and arguments
    is_pruning = True
    is_static = True
    #is_kd = True
    # kd_temperature = 4

    # RBGP variable configuration
    # osp_isp_choices = [0,0.5] #[[0,0.5],[0,0.75],[0,0.875],[0,0.9375]]
    # osp_isp_choices = [[0,0.9375]]
    # osp_isp_choices=[[0,0.25],[0,0.5],[0,0.75],[0,0.875],[0,0.9375]]
    osp_isp_choices=[[osp_size,isp_size]]
    # osp_isp_choices = [[0.75,0]] ##Row and column sparsity commands
    # block_config_choices=[[[-1,-1], [-1,-1], [2,2]],[[-1,-1], [-1,-1], [4,4]],[[-1,-1],[-1,-1],[8,8]]]
    block_config_choices=[[[-1,-1], [-1,-1], [2,2]]]
    # block_config_choices.append(None)
    # block_config_choices.append([[-1,-1], [-1,-1], [1,1]]) #[obh,obw],[cbh.cbw] #unstructured
    # block_config_choices.append([[1,-1], [-1,-1], [1,1]] ##Row sparsity)
    # block_config_choices.append([[-1,1], [-1,-1], [1,1]]) ##Column sparsity
    # block_config_choices.append([[-1,-1], [-1,-1], [8,8]]) ##Block sparsity with block size 8
    
    # block_config_choices.append([[-1,-1], [-1,-1], [2,2]]) ##Block sparsity with block size 2
    # block_config_choices.append([[-1,-1], [-1,-1], [4,4]]) ##Block sparsity with block size 4
    
    dataset_choices = ["cityscapes"]
    arch = "drn_d_54"
    #arch = "mobilenet_v2"
    kd_choices = [False] #True

    # Hyper parameters
    if arch == "resnet18" or "mobilenet_v2" or "drn_d_*":
        lr = 0.01
        batch_size = 12
        epochs = 500
    elif arch == "cifar_wrn_40_4":
        lr = 0.1
        batch_size = 128
        epochs = 200
    elif arch == "cifar_vgg19_bn":
        lr = 0.1
        batch_size = 256
        epochs = 160
    else:
        print("Hyper parameters for {} not provided".format(arch))
        exit(-1)

    import itertools
    exp_id = 0
    BASE_GPU_ID = 0
    NUM_GPUS = 4
    cur_sbps = []
    for exp_id,exp_config in enumerate(itertools.product(dataset_choices,
                                        block_config_choices,
                                        osp_isp_choices,
                                        kd_choices)):

        # Decoding the configuration
        dataset, block_config, oisp, is_kd = exp_config
        print(dataset)
        osp,isp = oisp
        dataset_dir = dataset_path_map[dataset]
        print(dataset_dir)
        exp_dump_dir = "sparse_experiments/{}_{}_{}".format(pruner_type, dataset, arch)
        base_model_path = "experiments/dense_{}_{}/model_best.pth.tar".format(dataset, arch)

        if not os.path.exists(exp_dump_dir):
            os.makedirs(exp_dump_dir)
            print(f"Path made {exp_dump_dir}")
        # Are we providing RBGP configuration from outside ?
        is_rbgp_outside = block_config is None

        # Experimernt information
        exp_info = pruner_type
        if is_rbgp_outside:
            #exp_info += "_rbgp"
            exp_info += "_rbgpcum"
        else:
            oblock_size, cblock_size, iblock_size = block_config
            exp_info += "_" + "{}".format(input_size)
            exp_info += "_" + "{}x{}".format(oblock_size[0], oblock_size[1])
            exp_info += "_" + "{}x{}".format(cblock_size[0], cblock_size[1])
            exp_info += "_" + "{}x{}".format(iblock_size[0], iblock_size[1])
        exp_info += "_" + "{:.2f}-{}".format(osp*100, opat)
        exp_info += "_" + "{:.2f}-{}".format(isp*100, ipat)
        
        if cross_prob is not None:
            ### Ramanujan related ####
            assert(opat == "RAMANUJAN" and ipat == "RAMANUJAN")
            exp_info += "_{:.2f}".format(cross_prob*100)
            if is_symmetric:
                exp_info += "_" + "symmetric"
            ##########################

        if collapse_tensor:
            exp_info += "_" + "collapse"
        if is_repetitive:
            exp_info += "_" + "repetitive"
        if is_kd:
            exp_info += "_kd"

        print(exp_info)
        # Name of the experiment
        ename =  "sparse_{}_{}_{}_{}".format(pruner_type, dataset, arch, exp_info)
        exp_dir = os.path.join(exp_dump_dir, ename)
        gpu_id = BASE_GPU_ID + exp_id%NUM_GPUS
        print(f"Ename {ename}")
        """
        print("{:7.3f}".format(extract_spectral_gap(exp_dir)), end=",")
        print("{:7.3f}".format(extract_accuracy(exp_dir)), end=",")
        if cross_prob == 0.5:
            print()
        continue
        """
        if args.dry_run:
            print("{} {}".format(gpu_id,ename))
            if os.path.exists(exp_dir):
                if os.path.exists(os.path.join(exp_dir, "checkpoint.pth.tar")):
                    print("Remove experiment directory {}".format(exp_dir))
                    print("rm -rf {}".format(exp_dir))
                    print("{:7.3f}".format(extract_accuracy(exp_dir)))
                else:
                    print("Cleaning up empty directory")
                    import shutil
                    shutil.rmtree(exp_dir)
            continue

        # Create experiment directory if does not exists.
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        else:
            if os.path.exists(os.path.join(exp_dir, "checkpoint.pth.tar")):
                print("Remove experiment directory {}".format(exp_dir))
                print("rm -rf {}".format(exp_dir))
                exit(-1)


        # Dumping config file into experiment directory
        pconfig_path = os.path.join(exp_dir, "config.json")
        print(f"Config path {pconfig_path}")
        if is_rbgp_outside:
            #cpath = "./rbgp_configs/{}_{:.2f}_{:.2f}.json".format(arch, osp*100, isp*100)
            cpath = "./rbgp_configs/{}_{:.2f}.json".format(arch, isp*100)
            import os
            os.system("cp {} {}".format(cpath, pconfig_path))
        else:
            print("In dump file")
            dump_config_file(dataset, arch, pruner_type, oblock_size, cblock_size, iblock_size, osp, opat,
                isp, ipat, is_repetitive, collapse_tensor, cross_prob, is_symmetric, pconfig_path, input_config)

        # Model compression args
        mc_args = ""
        # Pruning related
        if is_pruning:
            mc_args += " --mc_pruning --pr-base-model {} --pr_config_path {}".format(base_model_path, pconfig_path)
            if is_static:
                mc_args += "  --pr-static"

        # Knowledge distillatin related
        if is_kd:
            mc_args += " --mc-kd --kd-teacher {} --kd-temperature {}".format(base_model_path, kd_temperature)

        # Final command
        cmd = "python semantic_seg_multigpu.py train -d {} --dataset {} --arch {} --exp_dir {} {} --input_size {} --lr {} --epochs {} -b {} | tee {}/log.txt".\
                    format(dataset_dir, dataset, arch, exp_dir, mc_args, input_size, lr, epochs, batch_size, exp_dir)

        # Executing command
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(gpu_id)+cmd
        pretty_cmd = cmd.replace(" --"," \\\n\t --")+"\n"
        print(pretty_cmd)
        continue

        p = subprocess.Popen(cmd, shell=True)
        cur_sbps.append(p)

        if exp_id%NUM_GPUS == NUM_GPUS-1:
            exit_codes = [p.wait() for p in cur_sbps]
            cur_sbps = [] # Emptying the process list
