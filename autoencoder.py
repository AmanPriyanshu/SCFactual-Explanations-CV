import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

class MNIST_AEModel(torch.nn.Module):
	def __init__(self):
		super(MNIST_AEModel, self).__init__()
		self.enc_linear_1 = torch.nn.Linear(1*28*28, 526)
		self.enc_linear_2 = torch.nn.Linear(526, 128)
		self.dec_linear_1 = torch.nn.Linear(128, 526)
		self.dec_linear_2 = torch.nn.Linear(526, 1*28*28)
		self.activation = torch.nn.ReLU()
		self.activation_sigmoid = torch.nn.Sigmoid()
		self.flatten_layer = torch.nn.Flatten()

	def forward(self, x):
		og_shape = x.shape
		x = self.flatten_layer(x)
		x = self.enc_linear_1(x)
		x = self.activation(x)
		x = self.enc_linear_2(x)
		x = self.activation_sigmoid(x)
		x = self.dec_linear_1(x)
		x = self.activation(x)
		x = self.dec_linear_2(x)
		x = self.activation_sigmoid(x)
		return x.view(og_shape)

class ConstructClassifier:
	def __init__(self, dataset, dataset_val):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.x, self.y = dataset.T[1:].T, dataset.T[0]
		self.x = np.reshape(self.x, (len(dataset), 1, 28, 28))/255
		self.x_val, self.y_val = dataset_val.T[1:].T, dataset_val.T[0]
		self.x_val = np.reshape(self.x_val, (len(dataset_val), 1, 28, 28))/255
		self.model = MNIST_AEModel()
		self.model.to(self.device)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

	def train_single_epoch(self, epoch=0, batch_size=8):
		bar = tqdm(range(0, len(self.y), batch_size))
		running_loss = 0.0
		self.model.train()
		for batch_idx, i in enumerate(bar):
			batch_x = torch.from_numpy(self.x[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			self.optimizer.zero_grad()
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_x.float())
			loss.backward()
			self.optimizer.step()
			running_loss+=loss.item()
			bar.set_description(str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 3)}))
		bar.close()
		running_loss_val = 0.0
		self.model.eval()
		for batch_idx_val, i in enumerate(range(0, len(self.y_val), batch_size)):
			batch_x = torch.from_numpy(self.x_val[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_x.float())
			running_loss_val+=loss.item()
		print("Training Performance:", {"loss": round(running_loss/(batch_idx+1), 3)})
		print("Validation Performance:", {"loss": round(running_loss_val/(batch_idx_val+1), 3)})
		return running_loss/(batch_idx+1), running_loss_val/(batch_idx+1)

	def train(self, epochs):
		for epoch in range(epochs):
			self.train_single_epoch(epoch)
		return self.model

if __name__ == '__main__':
	df = pd.read_csv("./data/mnist_train.csv")
	df = df.values
	df_val = pd.read_csv("./data/mnist_val.csv")
	df_val = df_val.values
	cc = ConstructClassifier(df, df_val)
	model = cc.train(10)
	torch.save(model, "./output/ae_model.pt")