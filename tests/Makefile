AR = ar
CC = g++
C = gcc

OBJS_DIR = obj

INCS = -I../include -I../src -I../third_party   -I/usr/local/include  
DEFS = -DUSE_GPU 
LIB_PATH = -L/usr/local/lib -L../
LIB_LINK = -lkaffe -lprotobuf  

CFLAGS = -Wall $(DEFS) $(INCS)  
LDFLAGS = $(LIB_PATH) $(LIB_LINK) 

TARGET = im2col
LIB_SRCS = im2col.cpp

% : %.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean all
all: $(TARGET)
clean:
	rm -f $(TARGET)
	rm -rf $(OBJS_DIR)
