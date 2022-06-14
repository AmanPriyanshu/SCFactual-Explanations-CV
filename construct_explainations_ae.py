import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_classifier import MNISTModel
from autoencoder import MNIST_AEModel
from matplotlib import pyplot as plt

class CFXai:
	def __init__(self, image, target_y, model_path="./output/model.pt", ae_model_path="./output/ae_model.pt", lr=0.001):
		self.x_dash_img = torch.from_numpy(np.reshape(image, (1, 1, 28, 28))).float()
		self.x = self.x_dash_img.clone()
		self.target_y = target_y
		self.model = torch.load(model_path)
		self.ae_model = torch.load(ae_model_path)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.ae_model.to(self.device)
		self.x, self.x_dash_img = self.x.to(self.device), self.x_dash_img.to(self.device)
		self.ungrad_model()
		self.x_dash = self.ae_model.encoder(self.ae_model.flatten_layer(self.x_dash_img)).detach()
		self.prob_layer = torch.nn.Softmax(dim=1)
		self.lr = lr
		self.criterion = torch.nn.CrossEntropyLoss()

	def ungrad_model(self):
		for param in self.model.parameters():
			param.requires_grad=False
		for param in self.ae_model.parameters():
			param.requires_grad=False

	def generate_cf_single_iter(self):
		self.x_dash.requires_grad=True
		x_dash_img = self.ae_model.decoder(self.x_dash)
		x_dash_img = x_dash_img.view(1, 1, 28, 28)
		out = self.model(x_dash_img)
		pred = torch.argmax(out, dim=1)
		predicted = pred.item()
		loss = self.criterion(out, torch.tensor([self.target_y]).to(self.device))
		#loss = torch.square(1-pred.T[self.target_y][0])
		d = torch.mean(torch.abs(x_dash_img - self.x))
		# if d.item()!=0:
		# 	d.backward()
		loss.backward()
		self.x_dash = self.x_dash - self.lr*self.x_dash.grad
		self.x_dash = self.x_dash.detach()
		return loss.item(), d.item(), predicted==self.target_y

	def generate_cf(self, num_iters):
		bar = tqdm(range(num_iters))
		for epoch in bar:
			loss, d, stop = self.generate_cf_single_iter()
			bar.set_description(str({"epoch": epoch+1, "loss": round(loss, 3), "d": round(d, 3)}))
			# if stop:
			# 	print("Found Counterfactual at iteration="+str(epoch+1))
			# 	break
		bar.close()
		x_dash_img = self.ae_model.decoder(self.x_dash)
		x_dash_img = x_dash_img.cpu().numpy()[0]
		x_dash_img = np.reshape(x_dash_img, (28, 28))
		self.x = self.x.cpu().numpy()
		return x_dash_img, self.x[0][0], loss, d

if __name__ == '__main__':
	df_test = pd.read_csv("./data/mnist_test.csv")
	df_test = df_test.values

	first_image_xy = df_test[10]
	first_image_x = first_image_xy[1:]/255
	first_image_y = first_image_xy[0]
	print(first_image_y)
	cfxai = CFXai(first_image_x, target_y=8, lr=0.01)
	img, og_img, loss, d = cfxai.generate_cf(100)
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(og_img)
	ax1.set_title("Original Image")
	ax2.imshow(img)
	ax2.set_title("Counterfactual")
	plt.show()