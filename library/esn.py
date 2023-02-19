import numpy as np
import networkx as nx
class EchoState:
    def __init__(self, X, Y, N_x, gamma, beta):
        self.X = X
        self.Y = Y
        self.N_u = self.X.shape[1]
        self.N_y = self.Y.shape[1]
        self.N_x = N_x
        self.W_in = None
        self.W_out = None
        self.W = None
        self.r = np.random.uniform(size = (1, self.N_x))
        self.bias = np.random.uniform(size = (1, self.N_x))
        self.unit_bias = np.array([[1]])
        self.gamma = gamma
        self.beta = beta
    
    def _create_reservoir(self, **params):
        # create random matrix to construct reservoir
        random_matrix = np.random.uniform(low = -0.5, high = 0.5, size = (self.N_x, self.N_x))

        # generate graph
        if params['method'] == 'erdos-renyi':
            G = nx.erdos_renyi_graph(self.N_x, params['p'])
            A = np.array(nx.adjacency_matrix(G).todense())
            W = np.multiply(A, random_matrix)
            rho = np.amax(np.abs(np.linalg.eig(W)[0]))
            W = W / rho
            return W
        elif params['method'] == 'barabasi-albert':
            G = nx.barabasi_albert_graph(self.N_x, params['m'])
            A = np.array(nx.adjacency_matrix(G).todense())
            W = np.multiply(A, random_matrix)
            rho = np.amax(np.abs(np.linalg.eig(W)[0]))
            W = W / rho
            return W
        elif params['method'] == 'watts-strogatz':
            G = nx.watts_strogatz_graph(self.N_x, params['k'], params['p'])
            A = np.array(nx.adjacency_matrix(G).todense())
            W = np.multiply(A, random_matrix)
            rho = np.amax(np.abs(np.linalg.eig(W)[0]))
            W = W / rho
            return W
        else:
            print("Invalid method. Available methods are:\nerdos-renyi\nbarabasi-albert\nwatts-strogatz")
    
    def initiate_weights(self, **params):
        self.W_in = np.random.uniform(size = (1 + self.N_u, self.N_x))
        self.W_out = np.random.uniform(size = (1 + self.N_u + self.N_x, self.N_y))
        self.W = self._create_reservoir(**params)
        
    def _forward_pass(self, u, r):
        u = np.reshape(u, (1, -1))
        r = np.reshape(r, (1, -1))
        u_prime = np.concatenate((self.unit_bias, u), axis = 1) @ self.W_in
        new_r = (1 - self.gamma) * r + self.gamma * np.tanh(r @ self.W + u_prime + self.bias)
        return new_r
    
    def predict(self, u):
        u = np.reshape(u, (1, -1))
        self.r = self._forward_pass(u, self.r)
        return np.concatenate((self.unit_bias, u, self.r), axis = 1) @ self.W_out
    
    def fit(self):
        M = []
        for u in self.X:
            self.r = self._forward_pass(u, self.r)
            u = np.reshape(u, (1, -1))
            self.r = np.reshape(self.r, (1, -1))
            M.append(np.concatenate((self.unit_bias, u, self.r), axis = 1))

        M = np.squeeze(M)

        self.W_out = np.linalg.inv(M.T @ M + self.beta * np.identity(1 + self.N_u + self.N_x)) @ M.T @ self.Y