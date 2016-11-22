AR = ar
CC = g++
C = gcc

OBJS_DIR = obj

INCS = -I../include -I../src -I/usr/local/include  
DEFS = -DUSE_GPU 
LIB_PATH = -L/usr/local/lib -L../
LIB_LINK = -lkaffe -lprotobuf  

CFLAGS = -g -Wall $(DEFS) $(INCS)  
LDFLAGS = $(LIB_PATH) $(LIB_LINK) 

TARGET = demo
LIB_SRCS = demo.cpp 

LIB_OBJS = ${LIB_SRCS:%.cpp=$(OBJS_DIR)/%.o}

$(OBJS_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(LIB_OBJS)
	$(CC) -o $@ $(LIB_OBJS) $(LDFLAGS)

.PHONY: clean all
all: $(TARGET)
clean:
	rm -f $(TARGET)
	rm -rf $(OBJS_DIR)
