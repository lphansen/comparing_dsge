# Set this to "mpi" if you would like to use MPI, otherwise set this
# to "nompi".
VERSION=nompi


# The location of the Eigen C++ include files.
EIGEN_INCLUDE_DIR=$(HOME)/Git_projects/generalizedModel-Python/src/cppCore/eigen
PYTHON_INCLUDE_DIR=/software/python-3.5.2-el7-x86_64/include/python3.5m
PYBIND_INCLUDE_DIR=$(HOME)/.local/include/python3.5m

# These are the libraries used. (Note the order in which the libraries
# are listed is important.)
LDFLAGS=-lmkl_rt -lifcore -lpthread -lm \
-lboost_program_options -lboost_filesystem -lboost_system -lpython3.5m

BOOSTFLAGS=-lboost_filesystem -lboost_system

# C++ compiler settings.
CXXFLAGS=-DUSE_DOUBLES=1 -g -O3 -qopt-report=5 -std=gnu++0x -qopenmp -mcmodel=large \
-DBOOST_NO_CXX11_SCOPED_ENUMS -I$(EIGEN_INCLUDE_DIR) -I$(PYTHON_INCLUDE_DIR) -I$(PYBIND_INCLUDE_DIR) -I.
DEPS=parameters.h

# Source files used to build the program.
HEADERS=common.h derivs.h\
	matrixVars.h stateVars.h valueVars.h Vars.h parameters.h\
	functions.h model.h
SOURCES=common.cpp derivs.cpp\
	matrixVars.cpp stateVars.cpp valueVars.cpp Vars.cpp\
	functions.cpp model.cpp
#
#
# Object files to be built.
OBJS=main.o common.o Parameters.o derivs.o functions.o matrixVars.o valueVars.o Vars.o stateVars.o model.o

ifeq ($(VERSION),mpi)

	# Settings for the program that uses MPI.
	TARGET=longrunrisk__mpi
	CXX=mpiicpc
	CXXFLAGS+=-DUSE_MPI
else

	# Settings for the program that does not use MPI.
	TARGET=longrunrisk
	CXX=icc
endif

# RULES
# -----
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(TARGET)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
