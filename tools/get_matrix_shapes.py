import torch
import torchvision

import collections
import json

def forward_hook(self, input, output):
	print_info = False
	if type(self) in [torch.nn.Conv2d, torch.nn.Linear]:

		bs = input[0].shape[0]
		if type(self) == torch.nn.Conv2d:
			ofm, ifm = output.shape[1], input[0].shape[1]
			kh, kw = self.weight.shape[2], self.weight.shape[3]
			ih, iw = input[0].shape[2], input[0].shape[3]
			oh, ow = output.shape[2], output.shape[3]
			groups = self.groups

			M = ofm
			K = ifm*kh*kw
			N = bs*oh*ow
		elif type(self) == torch.nn.Linear:
			ofm, ifm = output.shape[1], input[0].shape[1]
			kh, kw = 1,1
			ih, iw = 1,1
			oh, ow = 1,1
			groups = 1

			M = ofm
			K = ifm*kh*kw
			N = bs
		else:
			print("Layer type {} is not supported".format(type(self)))
			exit(-1)

		# Calculating flops
		flops = bs * (ofm * oh * ow * (ifm//groups * kh * kw))
		self.flops = flops

		if print_info:
			fmt = "{:25s} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:10d}"
			print(fmt.format(self.name, ofm, ifm, kh, kw, ih, iw, oh, ow, groups, M, K, N, flops))

		# Creating layer information
		layer_info = collections.OrderedDict()
		layer_info["ofm"] = ofm
		layer_info["ifm"] = ifm
		layer_info["kh"] = kh
		layer_info["kw"] = kw
		layer_info["ih"] = ih
		layer_info["iw"] = iw
		layer_info["oh"] = oh
		layer_info["ow"] = ow
		layer_info["groups"] = groups

		layer_info["M"]  = M
		layer_info["K"]  = K
		layer_info["N"]  = N

		self.layer_info = layer_info

	else:
		print("Layer type {} is not supported".format(type(self)))
		exit(-1)

if __name__ == "__main__":
	"""
	arch = "vgg11_bn"
	model = torchvision.models.__dict__[arch]()
	"""
	from lmodels import cifar_rvgg11_512_bn
	model = cifar_rvgg11_512_bn()

	# Adding hooks to convolution and linear layers
	for name, module in model.named_modules():
		if type(module) in [torch.nn.Conv2d, torch.nn.Linear]:
			module.name = name
			module.register_forward_hook(forward_hook)

	# Running the model once to collect information
	input = torch.randn(1,3,32,32)
	output = model(input)

	# Constructing model_info 
	# NOTE : Exempting layers with groups != 1
	model_info = collections.OrderedDict()
	total_flops = 0
	selected_flops = 0
	for name, module in model.named_modules():
		if type(module) in [torch.nn.Conv2d, torch.nn.Linear]:
			if module.layer_info["groups"] == 1:
				model_info[name] = module.layer_info
				selected_flops += module.flops
			
			total_flops += module.flops

	print((1.0 - selected_flops/total_flops) * 100)

	json_data = json.dumps(model_info, indent=4)
	print(json_data)








