CXX_FLAGS=-O3 

ifeq (${MAKECMDGOALS},nvcc2)
SRC=$(wildcard tmp/*.cu)
OBJ=$(SRC:.cu=.o)
TYPE=cu 
else
SRC=$(wildcard tmp/*.cpp)
OBJ=$(SRC:.cpp=.o)
TYPE= cpp
endif

g++: 
	${MAKE} clean 
	${MAKE} gcc2
gcc2: CXX_FLAGS += -fopenmp  -march=native -Wno-attributes
gcc2: CXX=g++
gcc2: all

icpc: 
	${MAKE} clean 
	${MAKE} intel2
intel2: CXX_FLAGS += -openmp -xhost
intel2: CXX=icpc
intel2: all

knc: 
	${MAKE} clean 
	${MAKE} knc2
knc2: CXX_FLAGS += -openmp -mmic
knc2: CXX=icpc
knc2: all

avx512: 
	${MAKE} clean 
	${MAKE} avx5122
avx5122: CXX_FLAGS += -openmp -mmic 
avx5122: CXX=icpc
avx5122: all

nvcc:
	${MAKE} clean
	${MAKE} nvcc2
nvcc2: CXX_FLAGS = -rdc true -O3 -lgomp -Xcudafe "--diag_suppress=declared_but_not_referenced"
nvcc2: CXX = nvcc
nvcc2: all



mpi: 
	${MAKE} clean 
	${MAKE} mpi2
mpi2: CXX_FLAGS += -openmp -mavx 
mpi2: CXX=mpiicpc
mpi2: all


all: ${OBJ}
	${CXX} ${OBJ} ${LIB_PATH} ${LIBS} ${CXX_FLAGS} -o transpose.exe

%.o: %.${TYPE}
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf tmp/*.o transpose.exe
