# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-pthread")

# add_executable(alice_s alice_s.cpp nl2pc_ckks.cpp)
# target_link_libraries(alice_s PRIVATE SEAL::seal)

# add_executable(bob_c bob_c.cpp nl2pc_ckks.cpp)
# target_link_libraries(bob_c PRIVATE SEAL::seal)

pybind11_add_module(nl2pc nl2pc_ckks.cpp pybind_nl2pc.cpp)
target_link_libraries(nl2pc PRIVATE SEAL::seal)