CC := g++
CFLAGS := -std=c++11 -I/usr/include
LDFLAGS := -lEGL -lGLESv2

SRCS := main.cpp

OBJS := $(SRCS:.cpp=.o)
EXEC := matrix_multiplication_compute

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
