nvcc -arch=sm_70 -O3 main.cu \
     mpm_data.cu \
     mpm_kernels.cu \
     mpm_io.cpp \
     -o mpm_2d_gimp
