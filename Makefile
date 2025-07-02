# Neuroplus Neural Network Library Makefile

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++17 -O2 -g
INCLUDES = -Iinclude
LDFLAGS = -lm

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
TESTDIR = tests
EXAMPLEDIR = examples

# Source files and objects - Fixed pattern
LIB_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
EXAMPLE_SOURCES = $(wildcard $(EXAMPLEDIR)/*.cpp)
ALL_SOURCES = $(LIB_SOURCES) $(EXAMPLE_SOURCES)

LIB_OBJECTS = $(LIB_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:$(EXAMPLEDIR)/%.cpp=$(OBJDIR)/%.o)
ALL_OBJECTS = $(LIB_OBJECTS) $(EXAMPLE_OBJECTS)

EXECUTABLE = $(BINDIR)/neuroplus

# Test files
TEST_SOURCES = $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TESTDIR)/%.cpp=$(OBJDIR)/%.o)
TEST_EXECUTABLE = $(BINDIR)/test_runner

# Library target (only library sources, not examples)
LIBRARY = $(BINDIR)/libneuroplus.a

# Add dependency generation
DEPDIR = $(OBJDIR)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# Phony targets
.PHONY: all clean library tests run run-tests install help debug release info

# Default target
all: $(EXECUTABLE)

# Release build
release: CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++17 -O3 -DNDEBUG
release: clean $(EXECUTABLE)

# Debug build
debug: CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++17 -O0 -g -DDEBUG
debug: clean $(EXECUTABLE)

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# Compile source files with dependency generation
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile example files with dependency generation
$(OBJDIR)/%.o: $(EXAMPLEDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile test files with dependency generation
$(OBJDIR)/%.o: $(TESTDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(EXECUTABLE): $(ALL_OBJECTS) | $(BINDIR)
	$(CXX) $(ALL_OBJECTS) -o $@ $(LDFLAGS)

# Create static library (only library objects, not examples)
$(LIBRARY): $(LIB_OBJECTS) | $(BINDIR)
	ar rcs $@ $(LIB_OBJECTS)

# Build library target
library: $(LIBRARY)

# Build and run tests
tests: $(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(filter-out $(OBJDIR)/main.o, $(LIB_OBJECTS)) $(TEST_OBJECTS) | $(BINDIR)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Run main executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run tests
run-tests: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

# Install (basic implementation)
install: $(EXECUTABLE) $(LIBRARY)
	sudo cp $(EXECUTABLE) /usr/local/bin/
	sudo cp $(LIBRARY) /usr/local/lib/
	sudo mkdir -p /usr/local/include/neuroplus
	sudo cp -r $(INCDIR)/* /usr/local/include/neuroplus/

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Deep clean including backup files
clean-all: clean
	find . -name "*.bak" -o -name "*~" -o -name "*.swp" -o -name "*.d" | xargs rm -f

# Show help
help:
	@echo "Neuroplus Build System"
	@echo "======================"
	@echo "Targets:"
	@echo "  all        - Build main executable (default)"
	@echo "  release    - Build optimized release version"
	@echo "  debug      - Build debug version with symbols"
	@echo "  library    - Build static library"
	@echo "  tests      - Build test executable"
	@echo "  run        - Build and run main executable"
	@echo "  run-tests  - Build and run tests"
	@echo "  clean      - Remove build artifacts"
	@echo "  clean-all  - Deep clean including backup files"
	@echo "  install    - Install to system directories"
	@echo "  help       - Show this help message"

# Show build information
info:
	@echo "Build Configuration:"
	@echo "  Compiler: $(CXX)"
	@echo "  Flags: $(CXXFLAGS)"
	@echo "  Includes: $(INCLUDES)"
	@echo "  Library Sources: $(LIB_SOURCES)"
	@echo "  Example Sources: $(EXAMPLE_SOURCES)"
	@echo "  Library Objects: $(LIB_OBJECTS)"
	@echo "  Example Objects: $(EXAMPLE_OBJECTS)"

# Include dependency files
-include $(LIB_OBJECTS:.o=.d)
-include $(EXAMPLE_OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)