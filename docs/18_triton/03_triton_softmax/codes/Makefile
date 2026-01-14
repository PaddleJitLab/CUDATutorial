NVCC=nvcc
NVCC_FLAGS=-O3 -arch=native

TARGET=softmax
SRC=$(TARGET).cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
