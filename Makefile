AR = ar
CC = g++
C = gcc

OBJS_DIR = obj

INCS = -I./include -I./src -I./third_party  -I/usr/local/include  
DEFS = -DUSE_DEFAULT
LIB_PATH = -L/usr/local/lib
LIB_LINK = -lprotobuf 

CFLAGS = -MMD -MP -pthread -fPIC $(DEFS) $(INCS)  
LDFLAGS = $(LIB_PATH) $(LIB_LINK) 

TARGET = libkaffe.so
PROTOS = include/kaffe/proto/caffe.pb.cc src/proto/caffe.pb.h
LIB_SRCS = src/upgrade_proto.cpp\
		   src/blob.cpp\
		   src/layer.cpp\
		   src/net.cpp\
		   src/conv_layer.cpp\
		   src/fc_layer.cpp\
		   src/engine.cpp\
		   src/eigen_engine.cpp\
		   src/eigen_im2col.cpp

LIB_OBJS = ${LIB_SRCS:src/%.cpp=$(OBJS_DIR)/%.o}
LIB_OBJS += $(OBJS_DIR)/proto/caffe.pb.o

$(PROTOS) : src/caffe.proto
	@mkdir -p include/kaffe/proto
	protoc --proto_path=src --cpp_out=include/kaffe/proto $<

$(OBJS_DIR)/%.o : include/kaffe/%.cc
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJS_DIR)/%.o : src/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(PROTOS) $(LIB_OBJS)
	$(CC) -shared -o $@ $(LIB_OBJS) $(LDFLAGS)

.PHONY: clean all
all: $(TARGET)
clean:
	rm -rf include/kaffe/proto
	rm -f $(TARGET)
	rm -rf $(OBJS_DIR)
