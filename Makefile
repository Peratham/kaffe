AR = ar
CC = g++
C = gcc

OBJS_DIR = obj

INCS = -I./include -I./src  
DEFS = -DUSE_GPU 
LIB_PATH = -L/usr/local/lib
LIB_LINK = -lprotobuf 

CFLAGS = -g -Wall -fPIC $(DEFS) $(INCS)  
LDFLAGS = $(LIB_PATH) $(LIB_LINK) 

TARGET = libkaffe.so
PROTOS = src/proto/caffe.pb.cc src/proto/caffe.pb.h
LIB_SRCS = src/volume.cpp

LIB_OBJS = ${LIB_SRCS:src/%.cpp=$(OBJS_DIR)/%.o}
LIB_OBJS += $(OBJS_DIR)/proto/caffe.pb.o

$(PROTOS) : src/proto/caffe.proto
	@ echo PROTOC $<
	protoc --proto_path=src/proto --cpp_out=src/proto $<

$(OBJS_DIR)/%.o : src/%.cc
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
	rm -f $(PROTOS)
	rm -f $(TARGET)
	rm -rf $(OBJS_DIR)