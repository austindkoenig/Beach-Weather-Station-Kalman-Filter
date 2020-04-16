import numpy as np
import pandas as pd
import tqdm


class KalmanFilter:
    def __init__(self, dataframe, 
                state = None, 
                control_input = None, 
                state_covariance = None, 
                state_transition = None, 
                control_input_model = None, 
                observation_model = None, 
                process_noise_covariance = None, 
                observation_noise_covariance = None):
        self.dataframe = dataframe
        self.values = self.dataframe.to_numpy()
        self.n_iters, self.n_features = self.dataframe.shape

        # initialize state matrices
        self.state = state
        if self.state == None:
            self.state = np.random.normal(size = (self.n_features, 1))

        self.state_transition = state_transition
        if self.state_transition == None:
            self.state_transition = np.identity(self.n_features)

        self.state_covariance = state_covariance
        if self.state_covariance == None:
            self.state_covariance = np.diag(np.repeat(0.1, self.n_features))

        self.process_noise_covariance = process_noise_covariance
        if self.process_noise_covariance == None:
            self.process_noise_covariance = np.diag(np.repeat(1, self.n_features))

        self.control_input_model = control_input_model
        if self.control_input_model == None:
            self.control_input_model = np.identity(self.n_features)
        
        self.control_input = control_input
        if self.control_input == None:
            self.control_input = np.zeros((self.n_features, 1))
        
        # initialize measurement matrices
        self.observation_model = observation_model
        if self.observation_model == None:
            self.observation_model = np.identity(self.n_features)
        
        self.observation_noise_covariance = observation_noise_covariance
        if self.observation_noise_covariance == None:
            self.observation_noise_covariance = np.diag(np.repeat(4, self.n_features))

        self.state_estimates = [] # track list of state estimates
        self.kalman_gains = [] # track list of kalman gains to observe estimate-measurement relationship
        self.post_fit_residuals = [] # track the post-fit resuduals

    def predict(self):
        self.state = np.dot(self.state_transition, self.state) + np.dot(self.control_input_model, self.control_input)
        self.state_covariance = np.dot(self.state_transition, np.dot(self.state_covariance, self.state_transition.T)) + self.process_noise_covariance

    def update(self, t):
        predicted_measurement_mean = np.dot(self.observation_model, self.state)
        predicted_measurement_covariance = self.observation_noise_covariance + np.dot(self.observation_model, np.dot(self.state_covariance, self.observation_model.T))
        kalman_gain = np.dot(self.state_covariance, np.dot(self.observation_model.T, np.linalg.inv(predicted_measurement_covariance)))

        self.kalman_gains.append(kalman_gain)

        self.state = self.state + np.dot(kalman_gain, (self.values[t, :].reshape((self.n_features, 1)) - predicted_measurement_mean))

        self.state_estimates.append(self.state)

        self.state_covariance = self.state_covariance - np.dot(kalman_gain, np.dot(predicted_measurement_covariance, kalman_gain.T))
        post_fit_residuals = self.values[t, :].reshape((self.n_features, 1)) - np.dot(self.observation_model, self.state)

        self.post_fit_residuals.append(post_fit_residuals)


    def filter(self):
        for t in tqdm.tqdm(range(self.n_iters)):
            self.predict()
            self.update(t)
        
        self.state_estimates = pd.DataFrame(np.array(self.state_estimates).reshape((self.n_iters, self.n_features)), index = self.dataframe.index, columns = self.dataframe.columns)

        #self.kalman_gains = pd.DataFrame(np.array(self.kalman_gains).reshape((self.n_features, self.n_features, self.n_iters)), index = self.dataframe.index, columns = self.dataframe.columns)
        
        self.post_fit_residuals = pd.DataFrame(np.array(self.post_fit_residuals).reshape((self.n_iters, self.n_features)), index = self.dataframe.index, columns = self.dataframe.columns)