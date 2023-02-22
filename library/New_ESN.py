import numpy as np

class ESN:
    
    def __init__(self,input_scaling,epochs,N_u,N_y,N_r,sparsity = 0.2, alpha = 0.1, beta = 3, verbose = True, method = 'ridge_regression',TF = True):
        # Parameters

        self.input_scaling = input_scaling
        self.epochs = epochs # epochs
        self.N_u = N_u # input length
        self.N_y = N_y # output length
        self.N_r = N_r # reservoir size
        self.sparsity = sparsity # sparsity to resevoir to fast computing
        self.beta = beta  #coefficient for the regularization in the ridge regression
        self.x = np.zeros((N_r,1)) # the state of the echo state
        self.alpha = alpha # the leaky parameter for each iteration in the echo state
        self.verbose = verbose # show logs during the running of the deep echo
        self.method = method # method used for training the readout
        self.TF = TF # Teacher Forcing
        self.y_tf = 0
        
        # Initializing weights

        ## Initiate the W_in
        self.W_in = np.random.uniform(low=-self.input_scaling,high=self.input_scaling, size=(self.N_r,self.N_u))

        ## Initiate the reservoir weights
        self.W_reservoir = np.random.uniform(size=(self.N_r,self.N_r))
        self.W_reservoir[np.random.uniform(size=(self.N_r,self.N_r))<self.sparsity]=0

        ### Computing the spectral radius and applying
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_reservoir)))
        self.W_reservoir = self.W_reservoir/spectral_radius

        ## Iniciate the W_feedb weights
        self.W_feedb = np.random.uniform(size = (self.N_r, self.N_y))

        ## Initiate the W_out weights
        self.W_out = np.random.uniform(size = (self.N_y, 1 + self.N_u + self.N_r))

        if self.verbose:
            print('W_reservoir.shape: ', self.W_reservoir.shape)
            print('W_in.shape: ', self.W_in.shape)
            print('W_feedb.shape',self.W_feedb.shape)
            print('W_out.shape: ', self.W_out.shape)
        else:
            pass
    

    # Function to update the state of the reservoir
    def update(self,reservoir_state,input,output):
        ## Using teacher forcing
        if self.TF:
            reservoir_state = (1-self.alpha)*reservoir_state +\
            self.alpha*np.tanh(self.W_in @ input +
                                self.W_reservoir @ reservoir_state +
                                self.W_feedb @ np.reshape(output,(-1,1)))
    
        ## Without teacher forcing
        else:
            reservoir_state = (1-self.alpha)*reservoir_state +\
            self.alpha*np.tanh(self.W_in @ input + 
                               self.W_reservoir @ reservoir_state)
            
        return reservoir_state


    # Function to train the echo state
    def fit(self,X,Y):
        
        # Initiate the design matrix
        design_matrix = np.zeros((1 + self.N_u + self.N_r, X.shape[0]))

        ## iteration over epochs
        for epoch in range(self.epochs):

            self.y_tf = 0

            ## Iteration over the train data
            for i in range(X.shape[0]):
                self.x = self.update(self.x,X[i][:, np.newaxis],self.y_tf)
                self.y_tf = Y[i]
                ## Update the design matrix
                design_matrix[:, i] = np.squeeze(np.vstack([1,X[i][:, np.newaxis],self.x]))

            ## Update W_out
            if self.method == 'ridge_regression':
                self.W_out = Y.T @ design_matrix.T @ np.linalg.inv(\
                                                     design_matrix @ design_matrix.T +\
                                                     self.beta*np.identity(1 + self.N_u + self.N_r))
                
            if self.method == 'pseudo_inverse':
                self.W_out = Y.T @ np.linalg.pinv(design_matrix)
                
            if self.verbose:
                    print('Epoch', epoch + 1, 'of', self.epochs, end = '\r')


    # Function to make predictions
    def predict(self,test_input, test_label):

        predictions = []
        for u in range(len(test_input)):
            self.x = self.update(self.x,test_input[u][:, np.newaxis],self.y_tf)
            self.y_tf = test_label[u]
            design_matrix = np.squeeze(np.vstack([np.reshape(1,(-1,1)),
                                                test_input[u][:, np.newaxis],
                                                self.x]))
            predictions.append(self.W_out @ design_matrix)

            if self.verbose:
                print(f"Prediction {u}: {predictions[u]}")
        
        return predictions 
                

    