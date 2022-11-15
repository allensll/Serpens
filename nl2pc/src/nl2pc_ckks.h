#ifndef RELU_CKKS_H_
#define RELU_CKKS_H_

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <numeric>
#include <chrono>
#include <vector>
#include <thread>
#include <math.h>
#include "boost/asio.hpp"
#include "boost/asio/connect.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/asio/read.hpp"
#include "boost/asio/write.hpp"

// #include "pybind11/numpy.h"

#include "seal/seal.h"

using boost::asio::ip::tcp;


enum cs_role {
	ckksSERVER, ckksCLIENT
};

class NLParty {
    
    private:
        cs_role role;
        std::size_t nthreads_max;
        seal::SEALContext context;
        seal::SecretKey secret_key;
        seal::PublicKey public_key;
        seal::GaloisKeys galois_keys;
        seal::CKKSEncoder encoder;
        seal::Encryptor encryptor;
        seal::Decryptor decryptor;
        seal::Evaluator evaluator;
        std::size_t scale;
        bool cmpr;
        bool verbose;

        // io
        boost::asio::io_context io_context;
        boost::asio::io_context::work worker;
        tcp::socket socket;
        boost::asio::streambuf sb_w;
        boost::asio::streambuf sb_r;
        std::thread* th_ptr;

    public:
        NLParty(cs_role role,
            const std::string& address,
            const uint16_t port,
            uint32_t nthreads_max,
            seal::EncryptionParameters parms,
            seal::SecretKey secret_key,
            seal::PublicKey public_key,
            seal::GaloisKeys galois_keys,
            double scale,
            bool cmpr,
            bool verbose);

        ~NLParty();

        std::vector<double> cmp(const std::vector<double>& input);

        std::vector<double> relu(const std::vector<double>& input, bool rot);

        std::vector<double> maxpool2d(const std::vector<double>& input, const std::size_t patch_size, bool rot);
        
        // std::vector<double> ce_helper(std::vector<double>& input, std::vector<double>& k);

        // std::vector<double> mixup_ce_helper(std::vector<double>& input, std::vector<double>& k);

        bool is_server();
    
    private:

        void handler(boost::system::error_code ec, std::size_t length);

        void send_ciphers(std::vector<seal::Ciphertext>& en_data);

        void recv_ciphers(std::vector<seal::Ciphertext>& en_data);

        void encode_encrypt(const std::vector<double>& data,
            std::vector<seal::Ciphertext>& en_data,
            std::vector<std::size_t>& batchs);

        void decrypt_decode(const std::vector<seal::Ciphertext>& en_data,
            std::vector<double>& data,
            std::vector<std::size_t>& batchs);
        
        void ckks_cmp_helper(std::vector<seal::Ciphertext>& en_x1,
            const std::vector<double>& x2,
            const std::vector<double>& k,
            std::vector<std::size_t>& batchs);
        
        void ckks_nl_helper(std::vector<seal::Ciphertext>& en_a,
            std::vector<double>& k,
            const std::vector<double>& y2,
            std::vector<std::size_t>& batchs);

        void disorder(std::vector<seal::Ciphertext>& en_data,
            const std::vector<std::size_t>& s,
            std::vector<std::size_t>& batchs, std::size_t order);

        std::vector<std::size_t> shuffle(std::vector<seal::Ciphertext>& en_data, std::size_t patch_size);

        void reshuffle(std::vector<seal::Ciphertext>& en_data,
            std::vector<std::size_t>& order);

        void split_input(const std::vector<double>& input,
            std::vector<double>& input1,
            std::vector<double>& input2);
        
        void split_input2(const std::vector<double>& data, std::vector<double>& input1, std::vector<double>& input2, 
            std::size_t& pad_size1, std::size_t& pad_size2, std::size_t patch_size);

        std::vector<std::size_t> get_batchs(std::size_t n_data, std::size_t slot_count, std::size_t n_thread);
};

NLParty& create(cs_role role, const std::string& address, uint16_t port, uint32_t nthreads, bool cmpr, bool verbose);


void call_enc(const std::vector<double>& data,
        std::vector<seal::Ciphertext>& en_data,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale);

void call_dec(const std::vector<seal::Ciphertext>& en_data,
        std::vector<double>& data,
        seal::Decryptor& decryptor,
        seal::CKKSEncoder& encoder,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale);

void call_eval(std::vector<seal::Ciphertext>& en_x1,
        const std::vector<double>& x2,
        const std::vector<double>& k,
        seal::CKKSEncoder& encoder,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale);

void call_eval2(std::vector<seal::Ciphertext>& en_a,
        const std::vector<double>& k,
        const std::vector<double>& y2,
        seal::CKKSEncoder& encoder,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale);

void call_disorder(std::vector<seal::Ciphertext>& en_data,
        const std::vector<std::size_t>& s,
        const seal::GaloisKeys& galois_keys,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t order);

#endif
