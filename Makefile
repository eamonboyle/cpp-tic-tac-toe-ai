# Build tic-tac-toe on Linux/macOS (Visual Studio project remains under "Beginning C++ Game Programming/").
CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
SUB := Beginning\ C++\ Game\ Programming
TARGET := tictactoe

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SUB)/Beginning\ C++\ Game\ Programming.cpp
	cd $(SUB) && $(CXX) $(CXXFLAGS) Beginning\ C++\ Game\ Programming.cpp -o "$(CURDIR)/$(TARGET)"

clean:
	rm -f "$(TARGET)"
