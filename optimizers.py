import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as mp 
from sklearn.datasets import load_diabetes 
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm

X,y = load_diabetes(return_X_y=True)
#X,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

q = np.random.randn(x_train.shape[1])
class SGD():
	def __init__(self,x,y,q,epochs,loss):
		self.epochs = epochs
		self.eta = 0.01 #learning rate
		self.x = x ## data matrix
		self.weights = q## array of randints.
		self.bias = 0 ##
		self.y = y
		self.loss = loss

	def check(self):
		if self.loss == 'CE':
			self.GDn()
		elif self.loss == 'MSE':
			self.GD()
		else:
			print('shshs!')

	def y_hat(self,x):
		y_hat = self.weights@x+self.bias
		return y_hat

	def sigmoid(self,x):
		y_hat = (self.weights@x+self.bias)
		sigmoid = 1/(1+np.exp(-y_hat))
		return sigmoid

	def gradient_1(self,x,y,y_p):
		n = len(self.x)
		grad_w = -(x*(y-y_p))## returns array
		return grad_w

	def gradient_L(self,x,y,y_p):
		gradient_L = -x*(-y_p+y)
		return gradient_L

	def gradient_2(self,y,y_p):
		n = len(self.x)
		grad_b = -(y-y_p)*2 ## also returns array
		return grad_b

	def gradient_B(self,y,y_p):
		gradient_B = (-y_p+y)
		return gradient_B

	def GD(self):
		n = len(self.x)
		for i in tqdm(range(self.epochs)):
			for j in range(n):
				random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				#weights = self.weights[random_index]
				y_predict = self.y_hat(x_sample)
				# updation of weight and bias for every records
				self.weights = self.weights - self.eta * self.gradient_1(x_sample,y_sample,y_predict) 
				self.bias = self.bias - self.eta * self.gradient_2(y_sample,y_predict)  
		print('weights:',self.weights)
		print('next')
		print('Bias:',self.bias)

	def GDn(self):
		n = len(self.x)
		for i in tqdm(range(self.epochs)):
			for j in range(n):
				random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				#weights = self.weights[random_index]
				y_predict = self.sigmoid(x_sample)
				# updation of weight and bias for every records
				self.bias = self.bias - self.eta * self.gradient_B(y_sample,y_predict) 
				self.weights = self.weights - self.eta * self.gradient_L(x_sample,y_sample,y_predict)
		print('Bias:',self.bias)
		print('next')
		print('weights:',self.weights)  

#loss = 'CE'
#o = SGD(x_train,y_train,q,1000,loss)
#o.check()

class AdaGrad():
	def __init__(self,x,y,q,epochs,loss):
		self.epochs = epochs
		self.eta = 0.01 #learning rate
		self.x = x ## data matrix
		self.weights = q## array of randints.
		self.bias = 0 ##
		self.y = y
		self.loss = loss

	def check(self):
		if self.loss == 'CE':
			self.AdaGC()
		elif self.loss == 'MSE':
			self.AdaGM()
		else:
			print('shshs!')

	def y_hat(self,x):
		y =(self.weights@x+self.bias)
		return np.array(y)

	def sigmoid(self,x):
		#sigmoid = []
		#for i in range(len(x)):
		#y_hat =  -1*(self.weights@x+self.bias)
			#sigmoid_n = 1/(1+np.exp(y_hat))
		sigmoid = (1/(1+np.exp(-1*(self.weights@x+self.bias))))
		return np.array(sigmoid)

	def gradient_1(self,x,y,y_p):
		m = self.x.shape[0]
		s = -x*(y-y_p)
		return (2/m)*s

	def gradient_2(self,y,yp):
		s =-(y-yp)
		return (2/self.x.shape[0])*s

	def gradient_B(self,y,yp):
		s=(y-yp)
		return (2/self.x.shape[0])*s

	def gradient_L(self,x,y,y_p):
		s=-x*(y-y_p)
		return (2/self.x.shape[0])*s

	def AdaGM(self):
		n = len(self.x)

		for i in tqdm(range(self.epochs)):
			Gw = []
			Gb = []
			for j in range(n):
				#random_index=np.random.choice(self.x.shape[0],batch_size,replace=False)
				random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				y_predict = self.y_hat(x_sample)
				g = (self.gradient_1(x_sample,y_sample,y_predict))**2
				Gw.append(g)
				g_ = (self.gradient_2(y_sample,y_predict))**2
				Gb.append(g_)

				if len(Gw) == 1:
					self.weights = self.weights - self.eta * np.sqrt(Gw[0])
				elif len(Gw)>1:
					self.weights = self.weights - (self.eta/(np.sqrt(sum(Gw)+10**(-8)))) * self.gradient_1(x_sample,y_sample,y_predict)#Gw[len(Gw)-1]
				else:
					print('Error') 

				if len(Gb) == 1:
					self.bias = self.bias - self.eta * np.sqrt(Gb[0])
				elif len(Gb)>1:
					self.bias = self.bias - (self.eta/(np.sqrt(sum(Gb)+10**(-8)))) * self.gradient_2(y_sample,y_predict)
				else:
					print('Error')
		#print(len(Gw))
		print('weights:',self.weights)
		print('Bias:',self.bias)
		
	def AdaGC(self):
		n = len(self.x)

		for i in tqdm(range(self.epochs)):
			Gw = []
			Gb = []
			for j in range(n):
				#random_index=np.random.choice(self.x.shape[0],batch_size,replace=False)
				random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				y_predict = self.sigmoid(x_sample)
				g = (self.gradient_L(x_sample,y_sample,y_predict))**2
				Gw.append(g)
				g_ = (self.gradient_B(y_sample,y_predict))**2
				Gb.append(g_)

				if len(Gw) == 1:
					self.weights = self.weights - self.eta * np.sqrt(Gw[0])
				elif len(Gw)>1:
					self.weights = self.weights - (self.eta/(np.sqrt(sum(Gw)+10**(-8)))) * self.gradient_L(x_sample,y_sample,y_predict)#Gw[len(Gw)-1]
				else:
					print('Error') 

				if len(Gb) == 1:
					self.bias = self.bias - self.eta * np.sqrt(Gb[0])
				elif len(Gb)>1:
					self.bias = self.bias - (self.eta/(np.sqrt(sum(Gb)+10**(-8)))) * self.gradient_B(y_sample,y_predict)
				else:
					print('Error')
		print(len(Gw))
		print('weights:',self.weights)
		print('Bias:',self.bias)
#loss = 'CE'
#A = AdaGrad(x_train,y_train,q,1250,loss)
#A.check() 

class mini_batch():
	def __init__(self,x,y,q,epochs,loss):
		self.epochs = epochs
		self.eta = 0.01 #learning rate
		self.x = x ## data matrix
		self.weights = q## array of randints.
		self.bias = 0 ##
		self.y = y
		self.loss = loss

	def check(self):
		if self.loss == 'CE':
			self.miniC()
		elif self.loss == 'MSE':
			self.miniM()
		else:
			print('shshs!')

	def y_hat(self,x):
		y=[]
		for i in range(len(x)):
			y.append(self.weights@x[i]+self.bias)
		return np.array(y)

	def sigmoid(self,x):
		y = []
		for i in range(len(x)):
			y_hat = (self.weights@x[i]+self.bias)
			y.append(1/(1+np.exp(-y_hat)))
		return np.array(y)

	def gradient_1(self,x,y,y_p):
		s=0
		n=len(y)
		m = self.x.shape[0]
		for i in range(n):
			s+=-x[i]*(y[i]-y_p[i])
		return (2/m)*s

	def gradient_2(self,y,y_p):
		n=len(y)
		s=0
		for i in range(len(y)):
			s+=-(y[i]-y_p[i])
		return (2/self.x.shape[0])*s

	def gradient_B(self,y,y_p):
		s=0
		n=len(y)
		for i in range(n):
			s+=(y[i]-y_p[i])
		return (2/self.x.shape[0])*s

	def gradient_L(self,x,y,yp):
		s=0
		n=len(y)
		for i in range(n):
			s+=-x[i]*(y[i]-yp[i])
		return (2/x_train.shape[0])*s

	def miniM(self):
		n = len(self.x)
		batch_size = 100
		for i in tqdm(range(self.epochs)):

			for j in range(int(n/batch_size)):
				random_index=np.random.choice(self.x.shape[0],batch_size,replace=False)
				#random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				y_predict = self.y_hat(x_sample)
				# updation of weight and bias for every records
				self.weights = self.weights - self.eta * self.gradient_1(x_sample,y_sample,y_predict) 
				self.bias = self.bias - self.eta * self.gradient_2(y_sample,y_predict)  
		print('weights:',self.weights)
		print('Bias:',self.bias)

	def miniC(self):
		n = len(self.x)
		batch_size = 5
		for i in tqdm(range(self.epochs)):

			for j in range(int(n/batch_size)):
				random_index=np.random.choice(self.x.shape[0],batch_size,replace=False)
				#random_index = np.random.randint(0,n-1)
				x_sample = self.x[random_index]
				y_sample = self.y[random_index]
				y_predict = self.sigmoid(x_sample)
				# updation of weight and bias for every records
				self.weights = self.weights - (self.eta) * self.gradient_L(x_sample,y_sample,y_predict) 
				self.bias = self.bias - (self.eta) * self.gradient_B(y_sample,y_predict)  
		print('weights:',self.weights)
		print('Bias:',self.bias)
loss = 'MSE'		
m = mini_batch(x_train,y_train,q,1250,loss)
m.check() 