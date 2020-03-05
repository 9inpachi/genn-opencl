CXXFLAGS := -std=c++11 -Wall -Wpedantic -Wextra -MMD -MP `pkg-config --cflags OpenCL`
LINKFLAGS := `pkg-config --libs OpenCL`
OBJECTS := neuronUpdate.o init.o runner.o simulator.o
DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean

all: tutorial1

tutorial1: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@  $(LINKFLAGS)

-include $(DEPS)

%.d: %.cc
	@$(CXX) -M $(NVCCFLAGS) $< 1> $@

%.o: %.cc %.d
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	@rm -f $(OBJECTS) $(DEPS) tutorial1
