#include "nl2pc_ckks.h"

using boost::asio::ip::tcp;


NLParty& create(cs_role role, const std::string& address, uint16_t port, uint32_t nthreads, bool cmpr, bool verbose) {
    // ckks config
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    std::size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
    double scale = pow(2.0, 40);

    seal::SEALContext context(parms);
    seal::KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);     

    // insecurity !!!
    static NLParty party(role, address, port, nthreads, parms, secret_key, public_key, galois_keys, scale, cmpr, verbose);
    return party;
}


NLParty::NLParty(cs_role role,
        const std::string& address,
        const uint16_t port,
        uint32_t nthreads_max,
        seal::EncryptionParameters parms,
        seal::SecretKey secret_key,
        seal::PublicKey public_key,
        seal::GaloisKeys galois_keys,
        double scale,
        bool cmpr,
        bool verbose) :
            context(parms),
            encoder(context),
            encryptor(context, public_key, secret_key),
            decryptor(context, secret_key),
            evaluator(context),
            // io_context(),
            worker(io_context),
            socket(io_context) {

    this->role = role;
    this->nthreads_max = nthreads_max;
    this->scale = scale;
    this->cmpr = cmpr;
    this->verbose = verbose;

    this->th_ptr = new std::thread([this]() {this->io_context.run();});;
    this->th_ptr->detach();

    if (is_server()) {
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        socket = acceptor.accept();
    } else {
        tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(tcp::v4(), address, std::to_string(port)));
    }
    std::ostream o_stream(&sb_w);
    std::stringstream ss;
    auto size_key = galois_keys.save(ss, seal::compr_mode_type::none);
    o_stream.write(reinterpret_cast<const char*>(&size_key), sizeof(std::size_t));
    o_stream << ss.str();
    // send_size += boost::asio::write(socket, sb_w);
    boost::asio::async_write(socket, sb_w, std::bind(&NLParty::handler, this, std::placeholders::_1, std::placeholders::_2));
    ss.str("");

    boost::asio::read(socket, sb_r, boost::asio::transfer_exactly(sizeof(std::size_t)));
    std::size_t len;
    sb_r.sgetn(reinterpret_cast<char*>(&len), sizeof(std::size_t));
    boost::asio::read(socket, sb_r, boost::asio::transfer_exactly(len));
    ss << &sb_r;
    seal::GaloisKeys gk;
    gk.load(this->context, ss);
    ss.str("");
    this->galois_keys = gk;
}


NLParty::~NLParty() {
    io_context.stop();
    delete this->th_ptr;
    socket.close();
}


std::vector<double> NLParty::cmp(const std::vector<double>& data) {
    // spilt input data (example)
    // data.size = 15, solt_count = 4, nthreads_max = 2
    // [----------------------------------------------------------------------------------------------------------------------------------]

    // Server 1                                                                    Server 2
    // [------------------------------- input1 8  --------------------------------][--------------------- input2 7  ----------------------]

    // ciphertext 1                          ciphertext 2                          ciphertext 3                          ciphertext 4
    // [-------------- slot 4 --------------][-------------- slot 4 --------------][-------------- slot 4 --------------][---- slot 3 ----]

    // Server 1                                                                    Server 2
    // thread 1                              thread 2                              thread 1                              thread 2                       
    // [----------- ciphertext 1-- ---------][----------- ciphertext 2 -----------][----------- ciphertext 3 -----------][- ciphertext 4 -]

    // std::chrono::high_resolution_clock::time_point time_start, time_end;
    // std::chrono::microseconds time_diff;
    // time_start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> input1;
    std::vector<double> input2;
    split_input(data, input1, input2);

    std::size_t slot_count = encoder.slot_count();

    std::size_t n1 = input1.size();
    std::size_t n2 = input2.size();
    std::size_t n_cipher1 = ceil((double)n1 / slot_count);
    std::size_t n_cipher2 = ceil((double)n2 / slot_count);
    std::vector<seal::Ciphertext> en_data1(n_cipher1);
    std::vector<seal::Ciphertext> en_data2(n_cipher2);
    std::vector<double> res1(n1);
    std::vector<double> res2(n2);

    auto batchs1 = get_batchs(n1, slot_count, nthreads_max);
    auto batchs2 = get_batchs(n2, slot_count, nthreads_max);

    // time_end = std::chrono::high_resolution_clock::now();
    // time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    // std::cout << "Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;

    // step 1
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 1" << std::endl;}
        encode_encrypt(input1, en_data1, batchs1);
        send_ciphers(en_data1);
    }   
    // step 2
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 2" << std::endl;}
        std::vector<double> k(n2);
        for (std::size_t i=0; i<n2; i++) {
            // using rand() insecurity
            k[i] = static_cast<double>(rand() % 1000000 + 1) / 1000000;
        }

        recv_ciphers(en_data2);
        ckks_cmp_helper(en_data2, input2, k, batchs2);
        send_ciphers(en_data2);
    }
    // step 3
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 3" << std::endl;}

        recv_ciphers(en_data1);

        decrypt_decode(en_data1, res1, batchs1);
        for (std::size_t i=0; i<n1; i++) {
            res1[i] = (res1[i] > 0) ? 1 : 0;
        }
        boost::asio::async_write(socket, boost::asio::buffer(res1), boost::asio::transfer_exactly(sizeof(double)*n1));
    }
    // step 4
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 4" << std::endl;}
        boost::asio::read(socket, boost::asio::buffer(res2), boost::asio::transfer_exactly(sizeof(double)*n2));
    }
    // merge two vectors
    if (is_server()) {
        if (!input2.empty()) {
            res1.insert(res1.end(), res2.begin(), res2.end());
        }
        return res1;
    } else {
        if (!input1.empty()) {
            res2.insert(res2.end(), res1.begin(), res1.end());
        }
        return res2;
    }
}


std::vector<double> NLParty::relu(const std::vector<double>& data, bool rot) {
    
    std::vector<double> input1;
    std::vector<double> input2;
    std::size_t pad_size1;
    std::size_t pad_size2;
    split_input2(data, input1, input2, pad_size1, pad_size2, 1);
    // split_input(data, input1, input2);

    std::size_t slot_count = encoder.slot_count();

    std::size_t n1 = input1.size();
    std::size_t n2 = input2.size();
    std::size_t n_cipher1 = ceil((double)n1 / slot_count);
    std::size_t n_cipher2 = ceil((double)n2 / slot_count);
    std::vector<seal::Ciphertext> en_data1(n_cipher1);
    std::vector<seal::Ciphertext> en_data2(n_cipher2);
    std::vector<double> res1(n1);
    std::vector<double> res2(n2);

    auto batchs1 = get_batchs(n1, slot_count, nthreads_max);
    auto batchs2 = get_batchs(n2, slot_count, nthreads_max);

    // step 1
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 1" << std::endl;}
        encode_encrypt(input1, en_data1, batchs1);
        send_ciphers(en_data1);
    }   
    // step 2
    std::vector<std::size_t> s(n_cipher2);
    std::vector<seal::Ciphertext> en_step1_recv;
    std::vector<std::size_t> reorder;
    std::vector<double> k(n2);
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 2" << std::endl;}
        for (std::size_t i=0; i<n2; i++) {
            // using rand() insecurity
            k[i] = static_cast<double>(rand() % 1000000 + 1) / 1000000;
        }
        for (std::size_t i=0; i<n_cipher2; i++) {
            s[i] = rand() % slot_count - slot_count / 2;
        }

        recv_ciphers(en_data2);
        
        en_step1_recv = en_data2;
        ckks_cmp_helper(en_data2, input2, k, batchs2);
        // rotate en_data
        if (n_cipher2 < 2 || rot) {
            disorder(en_data2, s, batchs2, 1);
        }
        reorder = shuffle(en_data2, 1);
        send_ciphers(en_data2);
    }
    // step 3
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 3" << std::endl;}
       
        std::vector<double> a(n1);

        recv_ciphers(en_data1);

        decrypt_decode(en_data1, a, batchs1);
        for (std::size_t i=0; i<n1; i++) {
            a[i] = (a[i] > 0) ? a[i] : 0;
        }
        encode_encrypt(a, en_data1, batchs1);
        send_ciphers(en_data1);
    }
    // step 4
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 4" << std::endl;}
        recv_ciphers(en_data2);
        // re-rotate en_data
        reshuffle(en_data2, reorder);
        if (n_cipher2 < 2 || rot) {
            disorder(en_data2, s, batchs2, -1);
        }
        for (std::size_t i=0; i<n2; i++) {
            res2[i] = (double(rand() % 1000000) / 1000000 + 0.5) * input2[i];
        }
        ckks_nl_helper(en_data2, k, res2, batchs2);
        send_ciphers(en_data2);
    }
    // step 5
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 5" << std::endl;}
        recv_ciphers(en_data1);
        decrypt_decode(en_data1, res1, batchs1);
    }
    // merge two vectors
    res1.resize(n1 - pad_size1);
    res2.resize(n2 - pad_size2);
    if (is_server()) {
        if (!input2.empty()) {
            res1.insert(res1.end(), res2.begin(), res2.end());
        }
        return res1;
    } else {
        if (!input1.empty()) {
            res2.insert(res2.end(), res1.begin(), res1.end());
        }
        return res2;
    }
}


std::vector<double> NLParty::maxpool2d(const std::vector<double>& input, const std::size_t patch_size, bool rot) {
    
    // input.size = 100, patch_size = 4
    // [----------------------------------------------------------------------------------------------------------------------------------]

    // the number of pooling patch = 25
    // [-----------------------------------]
    // [-----------------------------------]
    // [-----------------------------------]
    // [-----------------------------------]

    // std::vector<double> input(input_.data(), input_.data() + input_.size());
    std::vector<double> input1;
    std::vector<double> input2;
    std::size_t pad_size1;
    std::size_t pad_size2;
    split_input2(input, input1, input2, pad_size1, pad_size2, patch_size);

    std::size_t slot_count = encoder.slot_count();


    std::size_t n1 = input1.size();
    std::size_t n2 = input2.size();
    std::size_t n1_ = n1 / patch_size;
    std::size_t n2_ = n2 / patch_size;

    std::size_t n_cipher1 = n1 / slot_count;
    std::size_t n_cipher2 = n2 / slot_count;
    std::vector<seal::Ciphertext> en_data1(n_cipher1);
    std::vector<seal::Ciphertext> en_data2(n_cipher2);
    std::vector<double> res1(n1);
    std::vector<double> res2(n2);

    if (this->verbose) {std::cout << "Len ipt1: " << n1 << "     Len ipt2: " << n2 << std::endl;}

    auto batchs1 = get_batchs(n1, slot_count, nthreads_max);
    auto batchs2 = get_batchs(n2, slot_count, nthreads_max);
    auto batchs1_ = get_batchs(n1_, slot_count, nthreads_max);
    auto batchs2_ = get_batchs(n2_, slot_count, nthreads_max);

    // step 1
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 1" << std::endl;}
        encode_encrypt(input1, en_data1, batchs1);
        send_ciphers(en_data1);
    }   
    // step 2
    std::vector<std::size_t> s(n_cipher2);
    std::vector<std::size_t> reorder;
    std::vector<double> k(n2);
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 2" << std::endl;}
        for (std::size_t i=0; i<n2_; i++) {
            // using rand() insecurity
            double rd = static_cast<double>(rand() % 1000000 + 500000) / 1000000;
            for (std::size_t j=0; j<patch_size; j++) {
                k[j*n2_+i] = rd;
            }
        }
        for (std::size_t i=0; i<n_cipher2/patch_size; i++) {
            double rd = rand() % slot_count - slot_count / 2;
            for (std::size_t j=0; j<patch_size; j++) {
                s[j*n_cipher2/patch_size+i] = rd;
            } 
        }
        recv_ciphers(en_data2);
        ckks_cmp_helper(en_data2, input2, k, batchs2);

        // rotate en_data
        if (n_cipher2 < 2 || rot) {
            disorder(en_data2, s, batchs2, 1);
        }
        reorder = shuffle(en_data2, patch_size);
        send_ciphers(en_data2);
    }
    // step 3
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 3" << std::endl;}
        std::vector<double> a(n1);

        recv_ciphers(en_data1);

        decrypt_decode(en_data1, a, batchs1);

        for (std::size_t i=0; i<n1_; i++) {
            std::size_t max_idx = i;
            for (std::size_t j=1; j<patch_size; j++) {
                if (a[max_idx] < a[j*n1_+i]) {
                    max_idx = j*n1_+i;
                }
            }
            a[i] = a[max_idx];
        }
        a.resize(n1_);
        en_data1.resize(n_cipher1/patch_size);
        // std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << std::endl;

        encode_encrypt(a, en_data1, batchs1_);
        send_ciphers(en_data1);
    }
    // step 4
    if (!input2.empty()) {
        if (this->verbose) {std::cout << "setp 4" << std::endl;}
        en_data2.resize(n_cipher2/patch_size);
        k.resize(n2_);
        recv_ciphers(en_data2);
        // re-rotate en_data
        reorder.resize(n_cipher2/patch_size);
        reshuffle(en_data2, reorder);
        s.resize(n_cipher2/patch_size);
        if (n_cipher2 < 2 || rot) {
            disorder(en_data2, s, batchs2_, -1);
        }
        for (std::size_t i=0; i<n2_; i++) {
            res2[i] = (double(rand() % 1000000) / 1000000 + 0.5) * input2[i];
        }
        ckks_nl_helper(en_data2, k, res2, batchs2_);
        send_ciphers(en_data2);
    }
    // step 5
    if (!input1.empty()) {
        if (this->verbose) {std::cout << "setp 5" << std::endl;}
        recv_ciphers(en_data1);
        decrypt_decode(en_data1, res1, batchs1_);
    }

    // merge two vectors
    res1.resize(n1_ - pad_size1);
    res2.resize(n2_ - pad_size2);
    if (is_server()) {
        if (!input2.empty()) {
            res1.insert(res1.end(), res2.begin(), res2.end());
        }
        return res1;
    } else {
        if (!input1.empty()) {
            res2.insert(res2.end(), res1.begin(), res1.end());
        }
        return res2;
    }
}


bool NLParty::is_server() {
    if (this->role == ckksSERVER) {
        return true;
    } else {
        return false;
    }
}


void NLParty::split_input(const std::vector<double>& data, std::vector<double>& input1, std::vector<double>& input2) {
    std::size_t slot_count = encoder.slot_count();
    std::size_t n_cipher = ceil((double)data.size() / slot_count);
    std::size_t len1 = ceil((double)n_cipher / 2) * slot_count;
    if (data.size() > slot_count) {
        if (is_server()) {
            input1.assign(data.begin(), data.begin()+len1);
            input2.assign(data.begin()+len1, data.end());
        } else {
            input2.assign(data.begin(), data.begin()+len1);
            input1.assign(data.begin()+len1, data.end());
        }
    } else {
        if (is_server()) {
            input1 = data;
        } else {
            input2 = data;
        }
    }
    if (this->verbose) {
        std::cout << "Length of input1: " << input1.size() << std::endl;
        std::cout << "Length of input2: " << input2.size() << std::endl;
    }
}


void NLParty::split_input2(const std::vector<double>& data, std::vector<double>& input1, std::vector<double>& input2, 
        std::size_t& pad_size1, std::size_t& pad_size2, std::size_t patch_size) {
    std::size_t slot_count = encoder.slot_count();
    std::size_t len_ = data.size() / patch_size;
    std::size_t n_cipher_ = ceil((double)len_ / slot_count);
    std::size_t lens_, pad_sizes, pad_sizec;
    std::size_t n_ciphers_ = ceil((double)n_cipher_ / 2);
    std::size_t n_cipherc_ = n_cipher_ - n_ciphers_;
    // std::cout << n_ciphers_ << " " << n_cipherc_ << std::endl;
    if (n_cipherc_ == 0) {
        lens_ = len_;
        pad_sizes = slot_count - len_;
        pad_sizec = 0;
    } else {
        lens_ = n_ciphers_ * slot_count;
        pad_sizes = 0;
        pad_sizec = n_cipherc_ * slot_count - (len_ - n_ciphers_ * slot_count);
    }
    // std::cout << pad_sizes << " " << pad_sizec << " " << lens_ << std::endl;
    for (std::size_t i=0; i<patch_size; i++) {
        std::size_t start = i * len_;
        if (is_server()) {
            input1.insert(input1.end(), data.begin()+start, data.begin()+start+lens_);
            input1.insert(input1.end(), pad_sizes, 0);
            input2.insert(input2.end(), data.begin()+start+lens_, data.begin()+start+len_);
            input2.insert(input2.end(), pad_sizec, 0);
            pad_size1 = pad_sizes;
            pad_size2 = pad_sizec;
        } else {
            input2.insert(input2.end(), data.begin()+start, data.begin()+start+lens_);
            input2.insert(input2.end(), pad_sizes, 0);
            input1.insert(input1.end(), data.begin()+start+lens_, data.begin()+start+len_);
            input1.insert(input1.end(), pad_sizec, 0);
            pad_size1 = pad_sizec;
            pad_size2 = pad_sizes;
        }
    }
    if (this->verbose) {
        std::cout << "Pad of input1: " << pad_size1 << "       Pad of input2: " << pad_size2 << std::endl;
        std::cout << "Length of input1: " << input1.size() << "       Length of input2: " << input2.size() << std::endl;
    }
}


void NLParty::handler(boost::system::error_code ec, std::size_t length) {
    if (this->verbose) {std::cout << "Send finished: " << (float)length/1024/1024 << "(MB)" << std::endl;}
    // std::cout << "Send finished: " << length << "(B)" << std::endl;

    if (ec) {
        std::cout << "Send error: " << ec.message() << std::endl;
    }
    return;
}


void NLParty::send_ciphers(std::vector<seal::Ciphertext>& en_data) {
    std::ostream o_stream(&sb_w);
    std::stringstream ss;
    for (std::size_t i=0; i<en_data.size(); i++) {
        std::size_t size_cipher;
        if (this->cmpr) {
            size_cipher = en_data[i].save(ss, seal::compr_mode_type::zstd);
        } else {
            size_cipher = en_data[i].save(ss, seal::compr_mode_type::none);
        }
        o_stream.write(reinterpret_cast<const char*>(&size_cipher), sizeof(std::size_t));
        o_stream << ss.str();
        ss.str("");
    }
    // send_size += boost::asio::write(socket, sb_w);
    boost::asio::async_write(socket, sb_w, std::bind(&NLParty::handler, this, std::placeholders::_1, std::placeholders::_2));
    return;
}


void NLParty::recv_ciphers(std::vector<seal::Ciphertext>& en_data) {
    std::stringstream ss;
    for (std::size_t i=0; i<en_data.size(); i++) {
        boost::asio::read(socket, sb_r, boost::asio::transfer_exactly(sizeof(std::size_t)));
        std::size_t len;
        sb_r.sgetn(reinterpret_cast<char*>(&len), sizeof(std::size_t));
        // std::cout << "recv: " << len << std::endl;
        boost::asio::read(socket, sb_r, boost::asio::transfer_exactly(len));
        ss << &sb_r;
        if (this->cmpr) {
            en_data[i].load(this->context, ss);
        } else {
            en_data[i].load(this->context, ss);
        }
        ss.str("");
    }
}


std::vector<std::size_t> NLParty::get_batchs(std::size_t n_data, std::size_t slot_count, std::size_t n_thread_max) {

    if (n_data == 0) {
        std::vector<std::size_t> batchs = {0};
        return batchs;
    }
    std::size_t last_cipher = n_data % slot_count;
    std::size_t n_cipher = ceil((double)n_data / slot_count);
    std::size_t n_thread = std::min(n_thread_max, n_cipher);

    std::vector<std::size_t> batchs(n_thread);

    std::size_t base = n_cipher / n_thread;
    std::size_t offset = n_cipher % n_thread;
    std::size_t len;
    
    for (std::size_t i=0; i<n_thread; i++) {
        len = base;
        if (i < offset) {
            len++;
        }
        len *= slot_count;
        if (last_cipher && (i == n_thread-1)) {
            len = len + last_cipher - slot_count;
        }
        batchs[i] = len;
    }
    return batchs;
}


void NLParty::encode_encrypt(const std::vector<double>& data,
        std::vector<seal::Ciphertext>& en_data,
        std::vector<std::size_t>& batchs) {

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    if (this->verbose) {
        time_start = std::chrono::high_resolution_clock::now();
    }

    std::size_t slot_count = encoder.slot_count();
    std::size_t n_data = data.size();
    std::size_t n_thread = batchs.size();
    std::vector<std::size_t> start_points(batchs.size());
    for (std::size_t i=1; i<start_points.size(); i++) {
        start_points[i] = start_points[i-1] + batchs[i-1];
    }

    std::thread t[n_thread];
    for (std::size_t i=0; i<n_thread; i++) {
        t[i] = std::thread(call_enc, std::cref(data), std::ref(en_data), std::ref(encoder), std::ref(encryptor), start_points[i], batchs[i], slot_count, scale);
    }
    for (std::size_t i=0; i<n_thread; i++) {
        t[i].join();
    }

    if (this->verbose) {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "Encryption Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }
}


void NLParty::decrypt_decode(const std::vector<seal::Ciphertext>& en_data,
        std::vector<double>& data,
        std::vector<std::size_t>& batchs) {
    
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    if (this->verbose) {
        time_start = std::chrono::high_resolution_clock::now();
    }

    std::size_t slot_count = encoder.slot_count();
    std::size_t n_data = std::accumulate(batchs.begin(), batchs.end(), 0.);
    std::size_t n_thread = batchs.size();

    std::vector<std::size_t> start_points(batchs.size());
    for (std::size_t i=1; i<start_points.size(); i++) {
        start_points[i] = start_points[i-1] + batchs[i-1];
    }

    std::thread t[n_thread];
    for (std::size_t i=0; i<n_thread; i++) {
        t[i] = std::thread(call_dec, std::cref(en_data), std::ref(data), std::ref(decryptor), std::ref(encoder), start_points[i], batchs[i], slot_count, scale);
    }
    for (std::size_t i=0; i<n_thread; i++) {
        t[i].join();
    }

    if (this->verbose) {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "Decryption Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }
}


void NLParty::ckks_cmp_helper(std::vector<seal::Ciphertext>& en_x1,
        const std::vector<double>& x2,
        const std::vector<double>& k,
        std::vector<std::size_t>& batchs) {
    
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    if (this->verbose) {
        time_start = std::chrono::high_resolution_clock::now();
    }

    std::size_t slot_count = encoder.slot_count();
    std::size_t n_data = x2.size();
    std::size_t n_thread = batchs.size();
    std::vector<std::size_t> start_points(batchs.size());
    for (std::size_t i=1; i<start_points.size(); i++) {
        start_points[i] = start_points[i-1] + batchs[i-1];
    }

    // method 2 add
    std::vector<double> x2_ = x2;
    for (std::size_t i=0; i<n_data; i++) {
        x2_[i] = x2_[i]*k[i];
    }
    // end method 2 add

    std::thread t[n_thread];
    for (std::size_t i=0; i<n_thread; i++) {
        t[i] = std::thread(call_eval, std::ref(en_x1), std::cref(x2_), std::cref(k), std::ref(encoder), std::ref(evaluator), start_points[i], batchs[i], slot_count, scale);
    }
    for (std::size_t i=0; i<n_thread; i++) {
        t[i].join();
    }
    if (this->verbose) {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "cmp_helper Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }
}


void NLParty::ckks_nl_helper(std::vector<seal::Ciphertext>& en_a,
        std::vector<double>& k,
        const std::vector<double>& y2,
        std::vector<std::size_t>& batchs) {

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    if (this->verbose) {
        time_start = std::chrono::high_resolution_clock::now();
    }

    std::size_t slot_count = encoder.slot_count();
    std::size_t n_data = k.size();
    std::size_t n_thread = batchs.size();
    std::vector<std::size_t> start_points(batchs.size());
    for (std::size_t i=1; i<start_points.size(); i++) {
        start_points[i] = start_points[i-1] + batchs[i-1];
    }
    for (std::size_t i=0; i<n_data; i++) {
        k[i] = 1/k[i];
    }

    std::thread t[n_thread];
    for (std::size_t i=0; i<n_thread; i++) {
        t[i] = std::thread(call_eval2, std::ref(en_a), std::cref(k), std::cref(y2), std::ref(encoder), std::ref(evaluator), start_points[i], batchs[i], slot_count, scale);
    }
    for (std::size_t i=0; i<n_thread; i++) {
        t[i].join();
    }
    if (this->verbose) {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "nl_helper Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }
}


void NLParty::disorder(std::vector<seal::Ciphertext>& en_data,
        const std::vector<std::size_t>& s,
        std::vector<std::size_t>& batchs, std::size_t order) {

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    if (this->verbose) {
        time_start = std::chrono::high_resolution_clock::now();
    }

    std::size_t slot_count = encoder.slot_count();
    std::size_t n_thread = batchs.size();

    std::vector<std::size_t> start_points(n_thread);
    std::vector<std::size_t> lens(n_thread);
    for (std::size_t i=1; i<n_thread; i++) {
        start_points[i] = start_points[i-1] + std::ceil((double)batchs[i-1] / slot_count);
    }
    std::thread t[n_thread];
    for (std::size_t i=0; i<n_thread; i++) {
        t[i] = std::thread(call_disorder, std::ref(en_data), std::cref(s), std::cref(this->galois_keys), std::ref(evaluator), start_points[i], std::ceil((double)batchs[i]/slot_count), order);
    }
    for (std::size_t i=0; i<n_thread; i++) {
        t[i].join();
    }
    if (this->verbose) {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "Rotation Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }
}


std::vector<std::size_t> NLParty::shuffle(std::vector<seal::Ciphertext>& en_data, std::size_t patch_size) {
    std::size_t n = en_data.size();
    std::size_t n_ = n / patch_size;
    std::vector<std::size_t> order(n);
    for (std::size_t i=0; i<n_; i++) {
        order[i] = i;
    }
    auto rng = std::default_random_engine {};
    std::shuffle(order.begin(), order.begin()+n_, rng);
    for (std::size_t i=1; i<patch_size; i++) {
        for (std::size_t j=0; j<n_; j++) {
            order[i*n_+j] = order[j] + i*n_;
        }
    }
    std::vector<std::size_t> reorder(n);
    for (std::size_t i=0; i<n; i++) {  
        reorder[order[i]] = i;
    }
    std::vector<seal::Ciphertext> output;
    std::transform(order.begin(), order.end(), std::back_inserter(output), [&](std::size_t i) { return en_data[i]; });
    en_data = output;
    return reorder;
}


void NLParty::reshuffle(std::vector<seal::Ciphertext>& en_data,
        std::vector<std::size_t>& reorder) {
    if (en_data.size() > 1) {
        std::vector<seal::Ciphertext> output;
        std::transform(reorder.begin(), reorder.end(), std::back_inserter(output), [&](std::size_t i) { return en_data[i]; });
        en_data = output;
    }
}


void call_enc(const std::vector<double>& data,
        std::vector<seal::Ciphertext>& en_data,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale) {

    std::size_t idx_cipher;
    std::size_t temp_len;
    for (std::size_t i=start_point; i<start_point+len; i+=slot_count) {
        temp_len = (i+slot_count < start_point + len) ? slot_count : start_point + len - i;
        std::vector<double> temp_data = {data.begin()+i, data.begin()+i+temp_len};
        seal::Plaintext temp_plain;
        seal::Ciphertext temp_encrypt;
        idx_cipher = (i+1) / slot_count;

        encoder.encode(temp_data, scale, temp_plain);
        encryptor.encrypt_symmetric(temp_plain, temp_encrypt);

        en_data[idx_cipher] = temp_encrypt;
    }
}


void call_dec(const std::vector<seal::Ciphertext>& en_data,
        std::vector<double>& data,
        seal::Decryptor& decryptor,
        seal::CKKSEncoder& encoder,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale) {
  
    std::size_t idx_cipher;
    for (std::size_t i=start_point; i<start_point+len; i+=slot_count) {
        std::vector<double> temp_data;
        seal::Plaintext temp_plain;
        idx_cipher = (i+1) / slot_count;
        decryptor.decrypt(en_data[idx_cipher], temp_plain);
        encoder.decode(temp_plain, temp_data);

        std::size_t temp_len = slot_count;
        if (i + slot_count > start_point + len) {
            temp_len = start_point + len - i;
        }
        std::copy(temp_data.begin(), temp_data.begin()+temp_len, data.begin()+i);
    }
}


void call_eval(std::vector<seal::Ciphertext>& en_x1,
        const std::vector<double>& x2,
        const std::vector<double>& k,
        seal::CKKSEncoder& encoder,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale) {

    std::size_t idx_cipher;
    std::size_t temp_len;
    for (std::size_t i=start_point; i<start_point+len; i+=slot_count) {
        temp_len = (i+slot_count < start_point + len) ? slot_count : start_point + len - i;
        std::vector<double> temp_x2 = {x2.begin()+i, x2.begin()+i+temp_len};
        std::vector<double> temp_k = {k.begin()+i, k.begin()+i+temp_len};
        seal::Plaintext temp_x2_plain, temp_k_plain;
        idx_cipher = (i+1) / slot_count;

        encoder.encode(temp_x2, scale, temp_x2_plain);
        encoder.encode(temp_k, scale, temp_k_plain);
        
        // method 1
        // // std::cout << "1" << std::endl;
        // // std::chrono::high_resolution_clock::time_point time_start, time_end;
        // // std::chrono::microseconds time_diff;
        // // time_start = std::chrono::high_resolution_clock::now();
        // evaluator.add_plain_inplace(en_x1[idx_cipher], temp_x2_plain);

        // // time_end = std::chrono::high_resolution_clock::now();
        // // time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        // // std::cout << "1 Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
        // // time_start = std::chrono::high_resolution_clock::now();
        // // std::cout << "1" << std::endl;

        // evaluator.multiply_plain_inplace(en_x1[idx_cipher], temp_k_plain);

        // // std::cout << "2" << std::endl;
        // // time_end = std::chrono::high_resolution_clock::now();
        // // time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        // // std::cout << "2 Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;

        // method 2
        evaluator.multiply_plain_inplace(en_x1[idx_cipher], temp_k_plain);
        evaluator.rescale_to_next_inplace(en_x1[idx_cipher]);
        en_x1[idx_cipher].scale() = pow(2.0, 40);
        evaluator.mod_switch_to_next_inplace(temp_x2_plain);
        evaluator.add_plain_inplace(en_x1[idx_cipher], temp_x2_plain);

        idx_cipher++;
    }
}


void call_eval2(std::vector<seal::Ciphertext>& en_a,
        const std::vector<double>& k,
        const std::vector<double>& y2,
        seal::CKKSEncoder& encoder,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t slot_count, std::size_t scale) {
    
    std::size_t idx_cipher;
    std::size_t temp_len;
    for (std::size_t i=start_point; i<start_point+len; i+=slot_count) {
        temp_len = (i+slot_count < start_point + len) ? slot_count : start_point + len - i;
        std::vector<double> temp_k = {k.begin()+i, k.begin()+i+temp_len};
        std::vector<double> temp_y2 = {y2.begin()+i, y2.begin()+i+temp_len};
        seal::Plaintext temp_k_plain, temp_y2_plain;
        idx_cipher = (i+1) / slot_count;

        encoder.encode(temp_k, scale, temp_k_plain);
        encoder.encode(temp_y2, scale, temp_y2_plain);

        evaluator.multiply_plain_inplace(en_a[idx_cipher], temp_k_plain);
        evaluator.rescale_to_next_inplace(en_a[idx_cipher]);
        en_a[idx_cipher].scale() = pow(2.0, 40);
        evaluator.mod_switch_to_next_inplace(temp_y2_plain);
        evaluator.sub_plain_inplace(en_a[idx_cipher], temp_y2_plain);

        idx_cipher++;
    }
}


void call_disorder(std::vector<seal::Ciphertext>& en_data,
        const std::vector<std::size_t>& s,
        const seal::GaloisKeys& galois_keys,
        seal::Evaluator& evaluator,
        std::size_t start_point, std::size_t len, std::size_t order) {

    for (std::size_t i=start_point; i<start_point+len; i++) {
        evaluator.rotate_vector_inplace(en_data[i], s[i]*order, galois_keys);
    }
}