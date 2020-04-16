import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

plt.style.use('ggplot')


data_dir = '../Data/Beach_Weather_Stations_-_Automated_Sensors.csv'
weather_df = pd.read_csv(data_dir)

weather_df = weather_df[~weather_df['Measurement Timestamp'].str.contains('2015')] # remove year 2015 because format is weird

weather_df['Measurement Timestamp'] = pd.to_datetime(weather_df['Measurement Timestamp'], infer_datetime_format = True)
weather_df.index = weather_df['Measurement Timestamp']
weather_df = weather_df.drop(['Measurement Timestamp'], axis = 1)

stations = weather_df['Station Name'].unique()

for s in stations:
    curr_df = weather_df[weather_df['Station Name'] == s][['Air Temperature', 
                                                           'Wet Bulb Temperature', 'Humidity', 
                                                           'Rain Intensity', 'Wind Speed', 
                                                           'Barometric Pressure', 'Solar Radiation', 
                                                           'Battery Life']]

    print(s)

    # intialize state matrices
    x = np.random.normal(size = (curr_df.shape[1], 1)) # state is 8x1 vector
    state_covariance = np.diag(np.repeat(0.1, curr_df.shape[1])) # state covariance matrix is 8x8 diagonal matrix of 0.1
    process_noise_covariance = np.diag(np.repeat(1, curr_df.shape[1])) # process noise covariance is 8x8 diagonal matrix of 1
    transition_model = np.identity(curr_df.shape[1]) # state transition model is 8x8 identity matrix
    control_model = np.identity(curr_df.shape[1]) # control input model is 8x8 identity matrix
    u = np.zeros(shape = (curr_df.shape[1], 1)) # control input is 8x1 vector of zeros, i.e. no control input

    # initialize measurement matrices
    observation_model = np.identity(curr_df.shape[1]) # observation model is 8x8 identity matrix
    observation_noise_covariance = np.diag(np.repeat(5, curr_df.shape[1])) # observation noise covariance is 8x8 diagonal matrix of 5

    n_iters = curr_df.shape[0]
    estimates = [x]

    for i in tqdm.tqdm(range(n_iters)):
        # predict
        x = np.dot(transition_model, x) + np.dot(control_model, u)
        state_covariance = np.dot(transition_model, np.dot(state_covariance, transition_model.T)) + process_noise_covariance

        # update
        predictive_measurement_mean = np.dot(observation_model, x)
        predictive_measurement_covariance = observation_noise_covariance + np.dot(observation_model, np.dot(state_covariance, observation_model.T))
        kalman_gain = np.dot(state_covariance, np.dot(observation_model.T, np.linalg.inv(predictive_measurement_covariance)))
        x = x + np.dot(kalman_gain, (curr_df.iloc[i, :].values.reshape((x.shape[0], 1)) - predictive_measurement_mean))
        state_covariance = state_covariance - np.dot(kalman_gain, np.dot(predictive_measurement_covariance, kalman_gain.T))
        measurement_post_fit_residuals = curr_df.iloc[i, :].values.reshape((x.shape[0], 1)) - np.dot(observation_model, x)

        estimates.append(x)
    
    estimates = pd.DataFrame(np.array(estimates[1:]).reshape((curr_df.shape[0], 8)), index = curr_df.index, columns = curr_df.columns)

    fig, ax = plt.subplots(figsize = (15, 10))

    ax.plot(range(len(curr_df['Air Temperature'][-1000:])), curr_df['Air Temperature'][-1000:], '--', label = 'Measured Air Temp')
    ax.plot(range(len(estimates['Air Temperature'][-1000:])), estimates['Air Temperature'][-1000:], ':', label = 'Estimated Air Temp')

    ax.legend()
    
    plt.show()
    break