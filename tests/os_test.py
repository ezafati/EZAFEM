import os.path
import numpy as np

path = os.path.dirname(__file__)
path = os.path.split(path)
print(path)

stiffmat = 10
mtype = 'stiff'
mat = eval(f'{mtype}mat')
print(mat)


class person:
    x = 1


a = person()


def myfunc(x):
    print(eval(f'x.x'))


A = np.ones((2,2))
b = np.array([1,3])

print(A.dot(b))
