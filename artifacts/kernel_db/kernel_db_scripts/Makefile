# Gencode arguments
# SMS ?= 30 35 37 50 52 60 61 70 75
SMS ?= 70

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

cc = nvcc ${GENCODE_FLAGS} -std=c++11 -I /usr/local/cuda/samples/common/inc $(LIBRARIES)

prom = profile
deps = kernel.cuh
src = $(shell find ./ -name "*.cu")
obj = $(src:%.cu=%.o)

$(prom) : $(obj)
	$(cc) -o $(prom) $(obj)

%.o : %.cu $(deps)
	$(cc) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(obj) $(prom)