import nl2pc
import random
import time
from contextlib import contextmanager

import numpy as np

random.seed(147)


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.2f} s'.format(text, time.perf_counter()-time_start))


def main():

    address = '127.0.0.1'
    port = 7766
    nthreads = 2
    
    n = 332667
    bob = nl2pc.Create(nl2pc.ckks_role.CLIENT, address=address, port=port, nthreads=nthreads, verbose=True)


    print("---CKKS relu---")
    for _ in range(3):
        _ = [random.random() for _ in range(n)]
        x = [random.random() for _ in range(n)]
        with timer():
            res = bob.relu(x)
        print(x[-3:])
        print(res[-3:])
    
    # print("---CKKS maxpool---")
    # for _ in range(3):
    #     _ = [random.random() for _ in range(n)]
    #     x = [random.random() for _ in range(n)]
    #     x = np.array(x)
    #     with timer():
    #         x = x.tolist()

    #     # x = [-1.1, 9.03, 4.01, -7, 0.3, 3, 9.1, 0]
    #     with timer():
    #         res = bob.maxpool2d(x, 9)
    #     print(x[24999], x[49999], x[74999], x[99999])
    #     print(res[24999])


if __name__ == "__main__":
    main()