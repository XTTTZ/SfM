import numpy as np

K=np.array([[1, 0, 0],
            [0, -1, 0],
            [0, 0, 2]])
S=np.array([1,2,3])
A=np.array([0,0,1])
E=np.dot(S,K)
E=np.dot(E,A)
print(E)
B=np.array([ [] ])
np.adj(B)
x = np.expm(np.arrary((2,2)))