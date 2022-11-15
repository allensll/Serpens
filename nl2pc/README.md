### build
---

* **Required Boost::ASIO, [pybind11](https://github.com/pybind/pybind11) and [SEAL](https://github.com/microsoft/SEAL)**

1. If necessary, download ABY, pybind11 and SEAL in `extern/`:
    ```
    git submodule add https://github.com/pybind/pybind11.git extern/pybind11
    ```
    ```
    git submodule add https://github.com/microsoft/SEAL.git extern/SEAL
    ```

2. Create and enter the build directory: `mkdir build && cd build`

3. Use CMake configure the build (depends on your gcc path):
    ```
    CC=/usr/bin/gcc-8 CXX=/usr/bin/g++-8 cmake ..
    ```
    This also initializes and updates the Git submodules of the dependencies
    located in `extern/`.

4. Call `make` in the build directory.
   You can find the libraries in the directories `build/src/cmpy.cpython-xxx-linux-gnu.so`.

### example
---

1. Copy `nl2pc.cpython-xxx-linux-gnu.so` into `example/`.

2. Open two terminals, run `python alice.py` and `python bob.py`, respectively.
