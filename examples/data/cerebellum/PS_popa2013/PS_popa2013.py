import numpy as np

m1 = np.genfromtxt('PSD_M1.csv', delimiter=',')
s1 = np.genfromtxt('PSD_S1.csv', delimiter=',')


with open('PSD_M1.npy', 'wb') as f:
    np.save(f, m1)

with open('PSD_S1.npy', 'wb') as f:
    np.save(f, s1)


# with open('PSD_M1.npy', 'rb') as f:
#     m1 = np.load(f)

# with open('PSD_S1.npy', 'rb') as f:
#     s1 = np.load(f)


# print(m1, s1)