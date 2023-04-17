import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load data from csv file
data = []
with open('/home/skanda/rss/DL-SLAM/PyICP-SLAM/result/02/pose02unoptimized_1681611872.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)

# Extract position and orientation information
positions = np.zeros((len(data), 3))
orientations = np.zeros((len(data), 3))
for i, row in enumerate(data):
    positions[i] = [float(row[1]), float(row[2]), float(row[3])]
    orientations[i] = [float(row[4]), float(row[5]), float(row[6])]

# Plot positions and orientations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2], 'b')
ax.quiver(positions[:,0], positions[:,1], positions[:,2], 
          np.cos(orientations[:,2]), np.sin(orientations[:,2]), np.zeros_like(orientations[:,2]), 
          color='r', length=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
