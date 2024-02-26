# Makefile for a C++ project

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -shared -fPIC -Wall

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Executable name
TARGET = $(BIN_DIR)/AtomicBasis.so

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony target to avoid conflicts with filenames
.PHONY: all clean

