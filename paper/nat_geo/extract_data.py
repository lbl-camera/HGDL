import numpy as np
a = np.genfromtxt("SRTM_RAMP2_TOPO_2000-02-11_rgb_3600x1800.CSV", delimiter = ",")
print(a.shape)
i = np.where(a == 99999.)
a[i] = 0.0
plt.imshow(a);plt.show()

map1 = a[420:620, 550:1050]
map1 = map1[::2,::2]   
plt.imshow(map1);plt.show()

data = np.empty((100*250,3))
counter = 0

for i in range(100):
    for j in range(250):
        data[counter,0] = i
        data[counter,1] = j
        data[counter,2] = map1[i,j]
        counter += 1

np.save("us_topo",data)

