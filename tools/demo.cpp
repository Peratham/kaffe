#include <string>

#include <kaffe/net.h>

typedef float Dtype;

int main(int argc, const char* argv[]) {
    const std::string param_file = argv[1];
    const std::string weight_file = argv[2];

    kaffe::Net<Dtype>* net = NULL;
    net = new kaffe::Net<Dtype>(param_file, weight_file);

    net->cpu();
}

