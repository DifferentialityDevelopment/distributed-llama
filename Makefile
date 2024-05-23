CXX = g++
CXXFLAGS = -std=c++11 -g -Werror -O3 -march=native -mtune=native
LDFLAGS = -lpthread

# Define a flag for Vulkan support
ifdef VULKAN
	LDFLAGS += -lvulkan
	CXXFLAGS += -DVULKAN
vulkan: src/vulkan.cpp
	$(CXX) $(CXXFLAGS) -c src/vulkan.cpp -o vulkan.o
endif

utils: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/utils.cpp -o utils.o
quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c src/quants.cpp -o quants.o
funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o
funcs-test: src/funcs-test.cpp funcs
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o
socket: src/socket.cpp
	$(CXX) $(CXXFLAGS) -c src/socket.cpp -o socket.o
transformer: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/transformer.cpp -o transformer.o
tasks: src/tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/tasks.cpp -o tasks.o
llama2-tasks: src/llama2-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/llama2-tasks.cpp -o llama2-tasks.o
grok1-tasks: src/grok1-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/grok1-tasks.cpp -o grok1-tasks.o
mixtral-tasks: src/mixtral-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/mixtral-tasks.cpp -o mixtral-tasks.o
tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o

ifdef VULKAN
main: src/main.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer vulkan
	$(CXX) $(CXXFLAGS) src/main.cpp -o main utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o vulkan.o ${LDFLAGS}
else
main: src/main.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/main.cpp -o main utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o ${LDFLAGS}
endif
funcs-test: src/funcs-test.cpp funcs utils quants
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o utils.o quants.o ${LDFLAGS}
quants-test: src/quants.cpp utils quants
	$(CXX) $(CXXFLAGS) src/quants-test.cpp -o quants-test utils.o quants.o ${LDFLAGS}
transformer-test: src/transformer-test.cpp funcs utils quants transformer socket
	$(CXX) $(CXXFLAGS) src/transformer-test.cpp -o transformer-test funcs.o utils.o quants.o transformer.o socket.o ${LDFLAGS}
llama2-tasks-test: src/llama2-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/llama2-tasks-test.cpp -o llama2-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o tokenizer.o ${LDFLAGS}
grok1-tasks-test: src/grok1-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/grok1-tasks-test.cpp -o grok1-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o tokenizer.o ${LDFLAGS}
vulkan-test: src/vulkan-test.cpp vulkan funcs utils quants
	$(CXX) $(CXXFLAGS) src/vulkan-test.cpp -o vulkan-test vulkan.o funcs.o utils.o quants.o ${LDFLAGS}
