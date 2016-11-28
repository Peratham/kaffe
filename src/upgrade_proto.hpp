#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <fcntl.h>
#include <unistd.h>
#include <ostream>
#include <iostream>
#include <string>
#include <assert.h>

#include <kaffe/proto/caffe.pb.h>
#include "common.hpp"

namespace caffe {

using std::string;
using std::map;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

// from io.cpp and io.hpp
inline bool ReadProtoFromTextFile(const std::string& fileString, Message* proto) {
    const char* filename = fileString.c_str();

    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << filename;
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
}

inline bool ReadProtoFromBinaryFile(const std::string& fileString, Message* proto) {
    const char* filename = fileString.c_str();

    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
}

// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const NetParameter& net_param);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                             V1LayerParameter* layer_param);

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

// Return true iff the Net contains input fields.
bool NetNeedsInputUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade input fields into layers.
void UpgradeNetInput(NetParameter* net_param);

// Return true iff the Net contains batch norm layers with manual local LRs.
bool NetNeedsBatchNormUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade batch norm layers.
void UpgradeNetBatchNorm(NetParameter* net_param);

// Return true iff the solver contains any old solver_type specified as enums
bool SolverNeedsTypeUpgrade(const SolverParameter& solver_param);

bool UpgradeSolverType(SolverParameter* solver_param);

// Check for deprecations and upgrade the SolverParameter as needed.
bool UpgradeSolverAsNeeded(const string& param_file, SolverParameter* param);

// Read parameters from a file into a SolverParameter proto message.
void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                       SolverParameter* param);

}  // namespace caffe


#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
