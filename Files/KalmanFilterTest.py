import pandas as pd
import KalmanFilter
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

    kf = KalmanFilter.KalmanFilter(dataframe = curr_df)
    kf.filter()

    fig, (temp_ax, err_ax) = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10))

    err_ax.plot(range(len(kf.post_fit_residuals['Humidity'][-500:])), kf.post_fit_residuals['Humidity'][-500:], color = 'red', label = 'Estimated Humidity Residuals')
    err_ax.legend()

    temp_ax.plot(range(len(curr_df['Humidity'][-500:])), curr_df['Humidity'][-500:], '--', label = 'Measured Humidity')
    temp_ax.plot(range(len(kf.state_estimates['Humidity'][-500:])), kf.state_estimates['Humidity'][-500:], ':', label = 'Estimated Humidity')
    temp_ax.legend()
    
    plt.show()

    break