SDK := $(shell xcrun --show-sdk-path)
CC  := xcrun clang

CFLAGS  := -O2 -fPIC -fobjc-arc -arch arm64 \
           -framework Metal -framework Foundation
LDFLAGS := -dynamiclib -install_name @rpath/libmetal_huffman.dylib

TARGET := libmetal_huffman.dylib

all: $(TARGET)

$(TARGET): metal_huffman.m metal_huffman.h huffman_kernels.metal
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ metal_huffman.m

clean:
	rm -f $(TARGET)

.PHONY: all clean
