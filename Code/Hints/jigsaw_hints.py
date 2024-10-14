

def constructor_1(self,a,b):
	self.Nodes = []
	for i in range(a):
		self.Nodes.append(b())
	self.OutputVector = np.zeros((a+1,)) ##augmentation occurs here. = \tilde{x}!
	self.OutputVector[0] = 1 #and there's my augmentation value

def constructor_2(self,b,c=1):
	self.Layers = [] ## initialised empty, added to via another function
	self.InputDimension = b
	self.OutputDimension = c



## the dot product should be a big giveaway as to what this function is doing!
def member_1(self,a):
	self.F = np.dot(self.A,a) #store in member for use later.
	return self.member_x(self.F)


## this is an initialiser -- a post-facto constructor
def member_2(self,a):
	#initialise vectors or correct shape
	self.Weights = np.random.normal(0,0.3,(a,))
	self.Gradient = np.zeros((a,))
	
	## only need these if you're using ADAM
	# self.C = np.zeros((a,))
	# self.D = np.zeros((a,))

	# #initialise counter
	# self.E = 1
	#initialise holder variables
	self.Y = 0
	self.dLdY = 0



def Predict(self,a): ## this is the predict module -- who does it belong to?
	if len(a) == self.B:
		a = np.insert(a,0,1) ## augment if needed. Provides option (when bulk-repeat processing, augmenting first will be better for speed)


	self.A[0].Predict(a)
	for i in range(1,len(self.A)):
		self.A[i].Predict(self.A[i-1].B)
	return self.A[-1].B[1:] ##have to omit the augment from the final ouput



# this updates the internal dLdY values, and adds to an internal gradient vector
def member_4(self, a, b): 
	self.dLdY = a
	self.Gradient += self.dF * b 


#this passes along an initialiser to its members. Maybe it's an initialiser itself?
def member_5(self,a):
	for b in self.A:
		b.Initialise(a)



def Predict(self, a):
	for i,b in enumerate(self.A):
		self.B[i+1] = b.member_x(a) ##offset for augmentation purposes!

def member_7(self,a,b):
	for i,c in enumerate(self.A):

		d= 0
		for e in a.A:
			d += c.A[i+1] * e.dF
		d *= c.member_x(c.F)
		c.member_x(d,b)

def member_8(self,a,b):
	for c in self.A:
		c.member_x(a * c.member_x(c.F),b)



def member_9(self,a):
	self.A.append(a)

def member_10(self):
	temp = self.B
	for a in self.A:
		a.member_x(temp)
		temp = len(a.A)
	if temp != self.C:
			print("ERROR! Output dimensions are inconsistent.")

def member_11(self,a):
	for b in self.A:
		b.member_x(a)

def member_12(self,a):
	step = a * self.B / np.linalg.norm(self.B)
	self.A += step
	self.B *= 0 ## clear for the next iteration!

def member_13(self,a): #default function (linear). Overwritten by child classes
	return a

def member_14(self,a):#default function (linear). Overwritten by child classes
	return 1

def Train(self, a,b,c ,d):
		self.member_x()

		e = 0.1
		N = len(a)
		for i in range(N):
			a[i]= np.insert(a[i],0,1) #augment the data up front

		for s in range(d):
			C = 0					
			for id in range(N): 
				p = self.member_x(a[id])
				C += c.member_x(p,b[id]) #call to populate .Fs 

				dC = c.member_x(p,b[id]) #call to populate .A[]

				self.A[-1].member_x(dC,self.A[-2].B)

				for j in np.arange(len(self.A)-2,-1,-1):
					if j > 0:
						self.A[j].member_x(self.A[j+1],self.A[j-1].B)
					else:
						self.A[j].member_x(self.A[j+1],a[id])
			for l in self.A:
				l.member_x(e)
			
			if s % 100 == 0:
				print(s,"C=",C,alpha)
				alpha*=0.9
		return

def member_16(self,a):
	return 1.0/(1.0 + np.exp(-a))

def member_17(self, a):
	return np.exp(-a)/(1.0 + np.exp(-a))**2


def member_18(self, a):
	if a > 0:
		return a
	return 0.01*a
def member_19(self, y):
	if a > 0:
		return 1
	return 0.01

def member_20(self,a,b):
	return -(a-b)**2

def member_21(self,a,b):
	return -2 * (a-b)