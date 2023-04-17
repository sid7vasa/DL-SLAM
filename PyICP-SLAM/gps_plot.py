import numpy as np
import glob

# Get a list of all the .npy files in the folder
folder_path = '/home/skanda/rss/DL-SLAM/PyICP-SLAM/DATA_2023-04-16_23-36-03/gps'
file_list = glob.glob(folder_path + '/*.npy')

# Initialize empty lists for x and y data
x_data = []
y_data = []

# Loop through each file and extract the x and y data
for file_path in file_list:
    npy_data = np.load(file_path, allow_pickle=True).item()
    x = npy_data['location']['x']
    y = npy_data['location']['y']
    x_data.append(x)
    y_data.append(y)

# Save the x and y data to a CSV file
with open('gps_data.csv', 'w') as file:
    file.write('x,y\n')
    for i in range(len(x_data)):
        file.write(str(x_data[i]) + ',' + str(y_data[i]) + '\n')

# Plot the x and y data
import matplotlib.pyplot as plt
plt.scatter(x_data, y_data)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
