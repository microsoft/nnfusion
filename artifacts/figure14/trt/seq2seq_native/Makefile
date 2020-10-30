SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
CUDA_TRIPLE=x86_64-linux
CUBLAS_TRIPLE=x86_64-linux-gnu
DLSW_TRIPLE=x86_64-linux-gnu
ifeq ($(TARGET), aarch64)
CUDA_TRIPLE=aarch64-linux
CUBLAS_TRIPLE=aarch64-linux-gnu
DLSW_TRIPLE=aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
CUDA_TRIPLE=aarch64-qnx
CUBLAS_TRIPLE=aarch64-qnx-gnu
DLSW_TRIPLE=aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), android64)
DLSW_TRIPLE=aarch64-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
CUBLAS_TRIPLE=$(DLSW_TRIPLE)
endif
export TARGET
export VERBOSE
export LIBDIR
export CUDA_TRIPLE
export CUBLAS_TRIPLE
export DLSW_TRIPLE
samples=sampleCharRNN seq2seq

# sampleMovieLensMPS should only be compiled for Linux targets.
# sample uses Linux specific shared memory and IPC libraries.
# ifeq ($(TARGET),x86_64)
# samples += sampleMovieLensMPS
# endif

.PHONY: all clean help test test_debug test_release
all:
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

test: test_debug test_release

test_debug:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_debug -C $(sample) &&) :

test_release:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_release -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\ttest[_debug|_release] - test all/release/debug samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
