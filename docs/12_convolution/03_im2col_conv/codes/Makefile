CC=nvcc

CXXFLAGS += -DNDEBUG -DUSE_DEFAULT_STDLIB -g

INCLUDES  += -I./include

LDFLAGS = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61, -gencode arch=compute_70,code=sm_70

# 获取当前目录下的cu文件集，放在变量CUR_SOURCE中
CUR_SOURCE=${wildcard ./src/*.cu}

# 将对应的cu文件名转为o文件后放在下面的CUR_OBJS变量中
CUR_OBJS=${patsubst %.cu, %.o, $(CUR_SOURCE)}

EXECUTABLE=conv2ddemo

all: $(EXECUTABLE)

$(EXECUTABLE): $(CUR_OBJS)
	$(CC) $(CUR_OBJS) $(LDFLAGS) -o $(EXECUTABLE)

%.o: %.cu
	$(CC) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ -Xptxas -v -lineinfo --std=c++11 ${LDFLAGS}

clean:
	rm -f $(EXECUTABLE)
	rm -f ./src/*.o