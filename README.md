# Code

### Requirements (our tested version)

* Ubuntu (20.04)
* Python (3.8)
* g++ (9.3)
* make (4.2.1)
* cmake (3.16.3)
* [PyTorch](https://pytorch.org/get-started/locally/) (1.10+cpu)
* [Scipy] (1.9.3)
    ```
    pip install scipy
    ```
* Boost::ASIO (1.71)
    ```
    sudo apt-get update
    sudo apt-get install libasio-dev
    ```
### Build

1. git clone https://github.com/allensll/Serpens2.git --recurse-submodules

2. Enter nl2pc directory: `cd nl2pc`

3. Create and enter the build directory: `mkdir build && cd build`

4. Configure and build: `cmake .. && make`
   You can find the library in the directories: build/src/nl2pc.cpython-xxx-linux-gnu.so.

### Example

1. Enter main directory: `cd ../..`

2. Copy `nl2pc.cpython-xxx-linux-gnu.so` into mian directory:
    `cp nl2pc/build/src/nl2pc.cpython-xxx-linux-gnu.so .`

#### Run two-server setting demo

3. Enter 2PC directory: `cd example/2PC`
    Open three terminals, run `python user.py`, `python server1.py` and `python server2.py` in example directory, respectively.

#### Run three-server setting demo

4. Enter 3PC directory: `cd example/3PC`.
    Open four terminals, run `python user.py`, `python server0.py`, `python server1.py` and `python server2.py` in example directory, respectively.


