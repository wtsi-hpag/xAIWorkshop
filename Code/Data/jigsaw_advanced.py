def member_advanced(self,alpha):

		b1 = 0.5
		b2 = 0.7
		c1 = 1.0/(1.0 - b1**self.L)
		c2 = 1.0/(1.0 - b2**self.L)
		self.E += 1
		
		self.B/=self.obfuscate 
		self.obfuscate = 0
		self.C = b1 * self.C + (1.0 - b1) * self.B
		self.D = b2 * self.D + (1.0 - b2) * self.B**2

		self.A -= alpha * (c1 *self.B)/np.sqrt(self.D*c2 + 1e-20)
		self.B *= 0 ## clear for the next iteration!

## This is clearly an advanced training function: how does it differ, what is it doing?
def member_advanced(self, data,labels,costFunc ,Nsteps):
		self.Initialise()

		alpha = 0.1
		prevC = None
		meanC = 0
		N = len(data)
		augmentedData = []
		for i in range(N):
			augmentedData.append(np.insert(data[i],0,1)) #augment the data up front

		if len(data) > 100:
			obfuscateSize = len(data)/100
		else:
			obfuscateSize = len(data)
		ids = np.arange(N)
		q =0
		for steps in range(Nsteps):
			np.random.shuffle(ids)

			obfuscate = 0
			sum = 0					
			for i in range(N): 
				id = ids[i]
				p = self.Predict(augmentedData[id])
				sum += costFunc.Compute(p,labels[id]) #have to call cost func first to populate node.Y (forward-pass)
				dCdP = costFunc.Gradient(p,labels[id]) #computes gradient with respect to the prediction
				self.Layers[-1].FinalLayerGradient(dCdP,self.Layers[-2].OutputVector)

				for j in np.arange(len(self.Layers)-2,-1,-1):
					if j > 0:
						self.Layers[j].MiddleLayerGradient(self.Layers[j+1],self.Layers[j-1].OutputVector)
					else:
						self.Layers[j].MiddleLayerGradient(self.Layers[j+1],augmentedData[id])
				obfuscate+=1
				if obfuscate == int(obfuscateSize+1):
					obfuscate = 0
					for layer in self.Layers:
						layer.UpdateWeights(alpha)
			if obfuscate!= 0:
				for layer in self.Layers:
					layer.UpdateWeights(alpha)
			meanC += sum
			q+=1
			obfuscateSize += 1
			if steps % 10 == 0:
				
				meanC /= q
				q = 0
				if prevC is not None:
					if meanC < prevC:
						print("Improved at",steps,"C=",meanC,obfuscateSize,alpha)
						alpha = min(0.2,alpha*1.1)
					else:
						print("Deproved at",steps,"C=",meanC,obfuscateSize,alpha)
						alpha = max(1e-5,alpha*0.7)
						
				prevC = meanC*1.0
			
		return