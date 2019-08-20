################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/convolution.cu 

OBJS += \
./src/convolution.o 

CU_DEPS += \
./src/convolution.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -G -g -O0 -gencode arch=compute_37,code=sm_37  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_37,code=compute_37 -gencode arch=compute_37,code=sm_37  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


