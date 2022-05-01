import argparse
# from matplotlib import image
import matplotlib
import numpy as np
from imageio import imread
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import os
# from os import join
import math
import cv2
import torchvision.transforms as T
import PIL.Image as Image
import json
#drn code imbibe
import drn
import data_transforms as transforms
# import wandb
# wandb.init()

CITYSCAPE_PALETTE = np.asarray([
	[128, 64, 128],
	[244, 35, 232],
	[70, 70, 70],
	[102, 102, 156],
	[190, 153, 153],
	[153, 153, 153],
	[250, 170, 30],
	[220, 220, 0],
	[107, 142, 35],
	[152, 251, 152],
	[70, 130, 180],
	[220, 20, 60],
	[255, 0, 0],
	[0, 0, 142],
	[0, 0, 70],
	[0, 60, 100],
	[0, 80, 100],
	[0, 0, 230],
	[119, 11, 32],
	[0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
	[0, 0, 0, 255],
	[217, 83, 79, 255],
	[91, 192, 222, 255]], dtype=np.uint8)


def fill_up_weights(up):
	w = up.weight.data
	f = math.ceil(w.size(2) / 2)
	c = (2 * f - 1 - f % 2) / (2. * f)
	for i in range(w.size(2)):
		for j in range(w.size(3)):
			w[0, 0, i, j] = \
				(1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
	for c in range(1, w.size(0)):
		w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
	def __init__(self, model_name, classes, pretrained_model=None,
				 pretrained=True, use_torch_up=False):
		super(DRNSeg, self).__init__()
		model = drn.__dict__.get(model_name)(
			pretrained=pretrained, num_classes=1000)
		pmodel = nn.DataParallel(model)
		if pretrained_model is not None:
			pmodel.load_state_dict(pretrained_model)
		self.base = nn.Sequential(*list(model.children())[:-2])

		self.seg = nn.Conv2d(model.out_dim, classes,
							 kernel_size=1, bias=True)
		self.softmax = nn.LogSoftmax()
		m = self.seg
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		m.bias.data.zero_()
		if use_torch_up:
			self.up = nn.UpsamplingBilinear2d(scale_factor=8)
		else:
			up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
									output_padding=0, groups=classes,
									bias=False)
			fill_up_weights(up)
			up.weight.requires_grad = False
			self.up = up

	def forward(self, x):
		x = self.base(x)
		x = self.seg(x)
		y = self.up(x)
		return self.softmax(y), x

	def optim_parameters(self, memo=None):
		for param in self.base.parameters():
			yield param
		for param in self.seg.parameters():
			yield param

# import imageio as iio
# from PIL import Image
def FrameCapture(video_path, model):
	vidObj = cv2.VideoCapture(video_path)
	# count = 0
	success = 1
	images = torch.zeros((100, 3, 300, 300))
	raw_images = []
	info = json.load(open('info.json'))
	normalize = transforms.Normalize(mean=info['mean'],
									 std=info['std'])

	i=0
	while success:
		success, image = vidObj.read()
		img = Image.fromarray(image, 'RGB')
		
		image_transform = T.Resize((300,300))
		img = image_transform(img)
		print("Image shape", img.size)
		raw_images.append(img)
		transform = T.Compose([transforms.ToTensorVideoImage(),
								normalize])# tran
		resized_img = transform(img)
		print(resized_img[0])
		images[i] = resized_img[0]
		i += 1
		if i == 100:
			break
		
	print(f"Shape of each frame is {images.shape}")
	images = torch.tensor(images, dtype=torch.float)
	
	return images

#     for i in range(len(images)):
#         start1=time.time()
#         color_image = Image.fromarray(CITYSCAPE_PALETTE[output[i].squeeze()])
#         inp_img = raw_images[i]
#         print('shapes ', inp_img.size, color_image.size)
#         # io.imsave("output",)

#         im1=ax1.imshow(inp_img)
#         im1.set_data(inp_img)
#         im2=ax1.imshow(color_image,alpha=0.6)
#         im2.set_data(color_image)

#         # figure.canvas.draw()
#         # figure.canvas.flush_events()
#         plt.savefig('pred_{}.png'.format(str(i)))
#         end2=time.time()
#         print("Image saved")

#         total_time=end2-start1
#         print(total_time)


# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='Segmentation demo with video')
#     parser.add_argument('--pretrained', 
#                         default="", 
#                         metavar='pretrained', help='path to pretrained weights.')
#     parser.add_argument('--inference', default=True, metavar='inference',
#                         help='To generate predictions on test set.')
#     parser.add_argument('--arch', default="drn_d_22")
#     parser.add_argument('-d', '--video_path', default=None, required=False)
#     parser.add_argument('-c', '--classes', default=19, type=int)
#     parser.add_argument('-s', '--crop-size', default=0, type=int)
#     parser.add_argument('--view', default=True, metavar='inference',
#                         help='View predictions at inference.')
#     parser.add_argument('--batch-size', type=int, default=200, metavar='N',
#                         help='input batch size for training (default: 200)')
#     parser.add_argument('--epochs', type=int, default=14, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--bn-sync', action='store_true')
#     parser.add_argument('--save-model', action='store_true', default=True,
#                         help='For Saving the current Model')

#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

from PyQt5.QtWidgets import *

# importing system
import sys

# importing numpy as np
import numpy as np
import cv2
# importing pyqtgraph as pg
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
import pyqtgraph.ptime as ptime

# Image View class
class ImageView(pg.ImageView):

	# constructor which inherit original
	# ImageView
	def __init__(self, *args, **kwargs):
		pg.ImageView.__init__(self, *args, **kwargs)


class Window(QMainWindow):

	def __init__(self):
		super().__init__()

		# setting title
		self.setWindowTitle("Semantic segmentation overlay")

		# setting geometry
		self.setGeometry(100, 100, 600, 500)

		# calling method
		self.UiComponents()

		# showing all the widgets
		self.show()

	# method for components
	def UiComponents(self):

		# creating a widget object
		widget = QWidget()

		# creating image view view object
		win = pg.GraphicsLayoutWidget()
 
		# adding view box object to graphic window
		view = win.addViewBox()
 
		##lock the aspect ratio so pixels are always square
		view.setAspectLocked(True)
 
		# Create image item
		self.img = pg.ImageItem(border='w')
		self.label = pg.ImageItem(border='w')
		# adding image item to the view box
		view.addItem(self.img)
		view.addItem(self.label)
		# self.img = ImageView()

		self.label.setZValue(1)

		# Set initial view bounds
		# view.setRange(QRectF(0, 0, 600, 600))

		# helps in incrementing
		self.i = 0
 
		# getting time
		self.updateTime = ptime.time()
                #print("Initial Time",self.updateTime)
		# fps
		self.fps = 0
 
		# method to update the data of image
		def updateData():
			
			# images = FrameCapture(video_path, model)

			model.eval()
			# for i in range(len(images)):
			# img = torch.unsqueeze(images, dim=0).cuda()
			# img=images.cuda()

			imagedata = images[self.i]
			numpydata = imagedata.cpu().detach().numpy()
			# print(numpydata.shape)
			# print(type(numpydata))
			numpydata = np.transpose(numpydata,(2,1,0))
			print("shape",imagedata.shape)
   			
			image = torch.unsqueeze(imagedata, 0)
			print("shape of input image",image.shape)

			# start1=time.time()
			final = model(image)[0]

			_, pred = torch.max(final, 1)
			output = pred.cpu().data.numpy()
			print("shape of pred", output.shape)
			# color_image = np.zeros((len(images), output.shape[1], output.shape[2],3))

			# for i in range(len(images)):
			color = Image.fromarray(CITYSCAPE_PALETTE[output.squeeze()])
			color_image = np.array(color)
			
			# print(color_image.shape)
			col_img = color_image
			# print("type of col img",type(col_img)
			print("shape of base image", numpydata.shape)
			print("shape of mask", col_img.shape)
			self.img.setImage(numpydata)
			#self.label.setImage(col_img, autoLevels=True, overlay=0.5)

			## Display the data
			# self.img.setImage(numpydata)
			print("inside",self.i)
			# creating new value of i
			self.i = (self.i + 1)
			if self.i>=40:
				return
			# creating a qtimer

			QTimer.singleShot(1, updateData)

			# getting current time
			now = ptime.time()
                        #print("The current time",now)

			# temporary fps
			fps2 = 1.0 / (now - self.updateTime)

			# updating the time
			self.updateTime = now

			# setting original fps value
			self.fps = self.fps * 0.9 + fps2 * 0.1
			# if self.i>150:
			# 	return

		# call the update method
		print("outside",self.i)
		updateData()

		# Creating a grid layout
		layout = QGridLayout()

		widget.setLayout(layout)
		# plot window goes on right side, spanning 3 rows
		layout.addWidget(win)

		# setting this widget as central widget of the main window
		self.setCentralWidget(widget)


arch = "drn_d_22"
classes = 19
pretrained = "drn_d_22_cityscapes.pth"
model = DRNSeg(arch, classes, pretrained_model=None,
					pretrained=False)
print("after model init")
# model = torch.nn.DataParallel(model)
print("original layers")
for params in model.state_dict():
	print(params)

if pretrained:
	model.load_state_dict(torch.load(pretrained))
print(model)

model = model
video_path = "mit_driveseg_sample.mp4"

images = FrameCapture(video_path, model)

# model.eval()
# final = model(images)[0]

# _, pred = torch.max(final, 1)
# output = pred.cpu().data.numpy()
# print("shape of pred", output.shape)
# color_image = np.zeros((len(images), output.shape[1], output.shape[2],3))
			
# for i in range(len(images)):
#     color = Image.fromarray(CITYSCAPE_PALETTE[output[i].squeeze()])
    
#     color_image[i] = np.array(color)
  
# print(color_image[1].shape)
# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())

