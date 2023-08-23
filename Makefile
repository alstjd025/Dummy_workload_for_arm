CC := g++
CFLAGS := -std=c++11 -I/usr/include
LDFLAGS := -lpthread -lEGL -lGLESv2

SRCS := dummy_workload.cpp main.cpp 

OBJS := $(SRCS:.cpp=.o)
EXEC := dummy_workload

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXEC)
