### Build

* **Required Boost::ASIO, [pybind11](https://github.com/pybind/pybind11) and [SEAL](https://github.com/microsoft/SEAL)**

1. If necessary, download pybind11 and SEAL in `extern/`:
    ```
    git submodule add https://github.com/pybind/pybind11.git extern/pybind11
    git submodule add https://github.com/microsoft/SEAL.git extern/SEAL
    ```
2. Create and enter the build directory: `mkdir build && cd build`

3. Configure and build: `cmake .. && make`
   You can find the library in the directories: build/src/nl2pc.cpython-xxx-linux-gnu.so.

### Example

1. Copy built library into example directory: `cp src/nl2pc.cpython-xxx-linux-gnu.so ../example/`

2. Open two terminals, run `python alice.py` and `python bob.py` in example directory, respectively.
