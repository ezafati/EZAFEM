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


A = np.ndarray((3,2,2))

print(A)
