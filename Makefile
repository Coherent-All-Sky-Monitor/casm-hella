
PROJECT = casm-hella

CUDA_ARCH ?= 89

SOURCE_PATH  ?= $(PWD)
BUILD_PATH   ?= $(PWD)/build
INSTALL_PATH ?= $(INST)/$(PROJECT)

# targets to configure for coverage, debug or release
.PHONY: configure-coverage configure-debug configure-release
configure-coverage: init
		@echo "===== START: configure-coverage ====="
		cd $(BUILD_PATH) && bash -c 'cmake $(SOURCE_PATH) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTS) $(DEPEND_OPTS) -DCMAKE_CXX_FLAGS="$(CXX_OPTS)" -DCODE_COVERAGE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
		cd $(BUILD_PATH) && bash $(SOURCE_PATH)/lint/sanitize_compile_commands.sh $(SOURCE_PATH) compile_commands.json
		@echo "=====  END: configure-coverage  ====="

configure-debug: init
	@echo "===== START: configure-debug ====="
	cd $(BUILD_PATH) && bash -c 'cmake $(SOURCE_PATH) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTS) $(DEPEND_OPTS) -DCMAKE_CXX_FLAGS="$(CXX_OPTS)" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
	cd $(BUILD_PATH) && bash $(SOURCE_PATH)/lint/sanitize_compile_commands.sh $(SOURCE_PATH) compile_commands.json
	@echo "=====  END: configure-debug  ====="
	@echo

configure-release: init
	@echo "===== START: configure-release ====="
	cd $(BUILD_PATH) && bash -c 'cmake $(SOURCE_PATH) -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTS) $(DEPEND_OPTS) -DCMAKE_CXX_FLAGS="$(CXX_OPTS)" -DBUILD_TESTING=OFF'
	@echo "=====  END: configure-release  ====="
	@echo

# targets to execute common init, clean, build, test and install operations
.PHONY: init clean build test install
init:
	@echo "===== START: init ====="
	rm -rf $(BUILD_PATH)
	mkdir -p $(BUILD_PATH)
	@echo "=====  END: init  ====="

clean:
	@echo "===== START: clean ====="
	cd $(BUILD_PATH) && bash -c 'make clean'
	@echo "=====  END: clean  ====="
	@echo

build:
	@echo "===== START: build ====="
	cd $(BUILD_PATH) && bash -c 'make -j$(PROC_COUNT)'
	@echo "=====  END: build  ====="
	@echo

test:
	@echo "===== START: test ====="
	cd $(BUILD_PATH) && bash -c 'ctest -T test --output-on-failure'
	@echo "=====  END: test  ====="
	@echo

install:
	@echo "===== START: install ====="
	cmake --install $(BUILD_PATH) $(INSTALL_OPTS)
	@echo "=====  END: install  ====="
	@echo

# targets for building for coverage, debug or release
.PHONY: build-coverage build-debug build-release
build-coverage: configure-coverage build
build-debug: configure-debug build
build-release: configure-release build

# targets for installing for with debug or release, noting that coverage installs are never done
.PHONY: install-debug install-release
install-debug: build-debug install
install-release: INSTALL_OPTS = --strip
install-release: build-release install
