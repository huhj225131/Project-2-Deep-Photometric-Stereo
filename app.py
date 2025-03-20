import cv2
import glob
import numpy as np
from sklearn.preprocessing import normalize

img = "cat"
#Load light source vector light.shape (3,n) n: number of light sources = number of images
light = np.loadtxt(f'./{img}/light_directions.txt')
light, light.shape

#Load image intensity and append to matrix I. I.shape (m, n) m: number of pixels, n: number of images 
folder_name = f'./{img}/Objects/'
ext = "png"
file_list = sorted(glob.glob(folder_name + "*." +ext))
I = None
height = 0
width =0
for file_name in file_list:
  im =  cv2.imread(file_name).astype(np.float64)
  im = np.mean(im, axis=2)
  height, width = im.shape
  if I is None:
    I = im.reshape((-1,1))
  else:
    I = np.append(I, im.reshape(-1, 1), axis = 1)

#compute each pixel's normal vector N.shape (m , 3)  m : number of pixels 
N = np.linalg.lstsq(light.T, I.T, rcond=None)[0].T
N_norm = normalize(N, axis = 1)
# N_norm = N

#compute albedo from N
albedo = np.linalg.norm(N,axis = 1)


normal_map = np.reshape(N_norm, (height, width, 3))

# Convert color channels (BGR <-> RGB) (đổi từ RGB sang BRG cho imshow hiển thị thoe BRGBRG)
normal_map = ((normal_map + 1) / 2 * 255).astype(np.uint8)
normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)  

# Visualize 
cv2.imshow("Normal Map", normal_map)
cv2.waitKey(0)
cv2.destroyAllWindows()