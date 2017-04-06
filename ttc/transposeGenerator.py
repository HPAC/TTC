# This script generates high-performance C/C++ code for any given multi-dimensional transposition.
#
# Tensor-Contraction Compiler (TTC), copyright (C) 2015 Paul Springer (springer@aices.rwth-aachen.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import ttc_util
import traceback
import itertools
import os
import copy
import math
import sys
import random

import transpose

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

###################################
#
# This file generates transpositions of the form B_perm(I) = alpha * A_I + beta * B_perm(I)
#
###################################

class transposeGenerator:
    def __init__(self, perm, loopPermutations, size, alpha, beta, maxNumImplementations,
            floatTypeA, floatTypeB, parallelize, streamingStores, prefetchDistances, blockings, papi,
            noTest, scalar, align, architecture, mpi, lda, ldb, silent,
            tmpDirectory, hotA = 0, hotB = 0, emitReference = 0):


        self.hotA = hotA
        self.hotB = hotB
        self.silent = silent

        self.tmpDirectory = tmpDirectory

        self.mpi = mpi 
        self.lda = copy.deepcopy(lda)
        self.ldb = copy.deepcopy(ldb)
        self.scalar = scalar
        self.noTest = noTest
        
        self.streamingStores = streamingStores
        self.architecture = architecture 
        self.cacheLineSize = 64 #in bytes
        self.alignmentRequirement = 32 #in bytes for AVX
        self.registerSizeBits = 256

        if architecture == "power":
            self.streamingStores = 0
            self.cacheLineSize = 128 #in bytes
        elif architecture == "knc":
            self.streamingStores = 0
            self.registerSizeBits = 512
            self.cacheLineSize = 128 #in bytes
            self.alignmentRequirement = 64
        elif architecture == "avx512":
            self.registerSizeBits = 512
            self.cacheLineSize = 128 #in bytes
            self.alignmentRequirement = 64

        self.parallelize = parallelize 
        self.floatTypeA = floatTypeA
        self.floatTypeB = floatTypeB
        self.papi = papi
        self.alpha = alpha
        self.beta = beta 
        self.size = copy.deepcopy(size)
        self.dim = len(perm)
        self.perm = copy.deepcopy(perm)
        self.indent = "   "

        self.prefetchDistances = copy.deepcopy(prefetchDistances)
        if( self.perm[0] == 0 ):
            self.prefetchDistances = [0] #we don't support prefetching in this case

        self.aligned = 1
        self.floatSizeA = ttc_util.getFloatTypeSize(floatTypeA)
        self.floatSizeB = ttc_util.getFloatTypeSize(floatTypeB)

        if( self.scalar == 1):
            self.microBlocking = ((1,1),"NOT AVAILABLE")
        else:
            self.microBlocking = self.getTranspositionMicroKernel()

        if(  self.microBlocking[0][0] * 8 * self.floatSizeA < self.registerSizeBits and self.floatTypeA != self.floatTypeB ):
           # this is not implemented yet => fallback to sclar
           self.scalar = 1
           self.microBlocking = (self.microBlocking[0],"NOT AVAILABLE")

        self.registerSizeBits = self.microBlocking[0][0] * 8 * self.floatSizeA

        #obey the alignment requirements for streaming-stores
        if( (self.size[0] * self.floatSizeA) % self.alignmentRequirement != 0 or
                (self.size[self.perm[0]] * self.floatSizeB) % self.alignmentRequirement != 0):
            self.aligned = 0
        if( align != 1 ):
            self.aligned = 0
        #initialize available blockings
        minA = self.microBlocking[0][0]
        minB = self.microBlocking[0][1]
        maxA = minA * 4
        maxB = minB * 4

        self.blockings = []
        if( not emitReference ):
            if( self.aligned != 1 and (self.architecture == "knc" or self.architecture == "power") ):
                print WARNING + "WARNING: non-aligned is not yet supported for the specified architecture" + ENDC
                print WARNING + "   => Fallback: use non-vectorized code." + ENDC
                self.scalar = 1

            if( self.perm[0] == 0): 
                if( len(blockings) == 0 ): #default, no blockings provided => use all blockings
                    if( len(self.size) == 1 ): #this is only the case if perm = IDENTITY _and_ lda and ldb are non-default
                        self.blockings.append((1,1))
                    else:
                        for i in range (1,11):
                            for j in range (1,11):
                                if( self.size[1] >= i and self.size[self.perm[1]] >= j ):
                                    self.blockings.append((i,j))
            else:
                if( len(blockings) == 0 ): #default, no blockings provided => use all blockings
                    for i in range (minA,maxA+1,minA):
                        for j in range (minB,maxB+1,minB):
                            if( self.size[0] >= i and self.size[self.perm[0]] >= j ):
                                self.blockings.append((i,j))
                else:
                    for blocking in blockings:
                        if( blocking[0] % minA != 0):
                            print FAIL + "[TTC] ERROR: blocking in A (%d) is not a multiple of %d."%(blocking[0],minA) + ENDC
                            exit(-1)
                        if( blocking[1] % minB != 0):
                            print FAIL + "[TTC] ERROR: blocking in B (%d) is not a multiple of %d."%(blocking[1],minB) + ENDC
                            exit(-1)

                        if( self.size[0] >= blocking[0] and self.size[self.perm[0]] >= blocking[1] ):
                            self.blockings.append(blocking)

            if( len(self.blockings) == 0): #this is needed to find solutions for which the size is smaller than the smallest blocking
                self.blockings.append((minA,minB))

            #sort blockings according to cost
            tmpBlockings = []
            for blocking in self.blockings:
                tmpBlockings.append((blocking, self.getCostBlocking(blocking)))
            
            tmpBlockings.sort(key=lambda tup: tup[1])
            tmpBlockings.reverse()

            self.blockings = []
            for (blocking, cost) in tmpBlockings:
                self.blockings.append(blocking)

        self.implementations = []
        self.maxNumImplementations = maxNumImplementations 

        
        #generate scalar version as reference
        optRef = ""
        if( ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
            optRef = "streamingstore"
        self.referenceImplementation = transpose.implementation((1,1),
                perm[-1::-1], self.perm, self.size, self.alpha, self.beta,
                self.floatTypeA, self.floatTypeB, optRef,  1, 0,(1,1),1, self.architecture, parallelize)

        self.minImplementationsPerFile = 64
        self.maxImplementationsPerFile = 256


        start = 0
        if( self.perm[0] == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
            start = 1
        if( len(loopPermutations) == 0):
            self.loopPermutations = []
            for loopPerm in itertools.permutations(range(start, self.dim)):
                self.loopPermutations.append(loopPerm)
        else:
            self.loopPermutations = copy.deepcopy(loopPermutations)

        # sort loopPermutations
        self.loopPermutations.sort(key=lambda tup: ttc_util.getCostLoop(tup, self.perm, self.size))

        ######################################
        # Reduce search space
        ######################################

        # combine the best sqrt(maxNumImplementations) blockings with the best sqrt() loopOrders
        # only keep the best sqrt(maxNumImplementations) blockings
        maxBlockings = math.ceil(math.sqrt(float(maxNumImplementations)))
        while( len(self.blockings) > maxBlockings 
                and len(self.blockings) * len(self.loopPermutations) > maxNumImplementations):
            self.blockings.pop()

        maxLoopPermutations= maxBlockings
        while( len(self.loopPermutations) > maxLoopPermutations
                and len(self.blockings) * len(self.loopPermutations) > maxNumImplementations):
            self.loopPermutations.pop()


    def getCostBlocking(self, blocking):
       if( len(self.size) == 1):
           return 1 #we don't have any blockings in this case
       #remainder should be zero
       size0 = self.size[0]
       sizep0 = self.size[self.perm[0]]
       if(self.perm[0] == 0):
           size0 = self.size[1]
           sizep0 = self.size[self.perm[1]]
       remainderA = (size0 % blocking[0]) / float(size0) #should be (close to) zero
       remainderB = (sizep0 % blocking[1]) / float(sizep0) #should be (close to) zero

       #blocking should be multiple of cacheline
       numElementsPerCacheLine = self.cacheLineSize / self.floatSizeA
       numCacheLines = (blocking[0] + numElementsPerCacheLine - 1) / numElementsPerCacheLine
       cacheLineUtilizationA = blocking[0] / float(numCacheLines * numElementsPerCacheLine) #should be (close to) one
       numCacheLines = (blocking[1] + numElementsPerCacheLine - 1) / numElementsPerCacheLine
       cacheLineUtilizationB = blocking[1] / float(numCacheLines * numElementsPerCacheLine) #should be (close to) one

       metric = (cacheLineUtilizationA + cacheLineUtilizationB)/2.0  #should be close to 1
       metric += ((1 - remainderA) + (1- remainderB))/2.0
       return metric /2.0

    def getNumSolutions(self):
        return len(self.implementations)

    def generateOffsetFile(self, directory):
        codeOffset = "#ifndef _TTC_OFFSET_H\n"
        codeOffset += "#define _TTC_OFFSET_H\n"
        codeOffset += "struct Offset\n{\n"
        codeOffset += "   int offsetA;\n"
        codeOffset += "   int offsetB;\n"
        codeOffset += "};\n\n"
        codeOffset += "#endif\n"
        if(directory[-1] != '/'):
           directory += '/'
        offsetFile = open(directory + "ttc_offset.h","w")
        offsetFile.write(codeOffset)
        offsetFile.close()

    def generateVersion(self,versionStr):
        #used to generate a specific implementation
        for impl in self.implementations:
            if( impl.getVersionName() == versionStr ):
                prefetchDistances = list(set([impl.getPrefetchDistance(), 0])) #we need prefetch distance 0 for the remainder while-loop
                code = "#if defined(__ICC) || defined(__INTEL_COMPILER)\n"
                code += "#define INLINE __forceinline\n"
                code += "#elif defined(__GNUC__) || defined(__GNUG__)\n"
                code += "#define INLINE __attribute__((always_inline))\n"
                code += "#endif\n\n"
                if(len(prefetchDistances)>1):
                    code += "#include <queue>\n"
                    code += "#include \"ttc_offset.h\"\n"
                    
                code += self.generateTranspositionKernel([impl.getBlocking()],prefetchDistances, 1, [impl.optimization])[0]
                return code + impl.getImplementation(self.parallelize, clean=1 )
        return ""

    def generate(self):
        if( len(self.size) != 1 and ( not(self.perm[0] == 0 and self.perm[1] == 1)) ): #only use code generator if at least one of the first two indices changes
            self.genCandidates()
        self.printMain()
        self.generateImplementations() 

    def getAppropriateOptimizations(self):
        optimizations = []

        # streaming-stores
        if( ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
            optimizations.append("streamingstore")
        else:
            optimizations.append("")

        return optimizations

    def listToString(self, perm):
        string = ""
        for s in perm:
            string += str(s) + ","
        print string
        return string[:-1]

    def genCandidates(self):

        optimizations = self.getAppropriateOptimizations()

        counter = 0
        #generate all implementations
        for prefetchDistance in self.prefetchDistances:
            for blocking in self.blockings:
                for loopPerm in self.loopPermutations:
                    
                    for opt in optimizations:
                    
                        if( opt == "streamingstore" and not ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
                            #skip this optimization if the blocking in B is not a multiple of the cacheLineSize
                            continue

                        counter += 1
                        if( self.silent != 1):
                            sys.stdout.write("[TTC] Implementations generated so far: %d                    \r"%counter)
                        sys.stdout.flush()
                        implementation = transpose.implementation(blocking, loopPerm,
                                self.perm, self.size, self.alpha, self.beta, self.floatTypeA, self.floatTypeB,
                                opt, self.scalar, prefetchDistance, self.microBlocking[0],
                                0, self.architecture, self.parallelize)

                        if( len(self.implementations) < self.maxNumImplementations ):
                            self.implementations.append(implementation)
                            self.implementations.sort(key=lambda tup: tup.getCostLoop() )
                        elif( self.implementations[-1].getCostLoop() > implementation.getCostLoop() ):
                            self.implementations.pop()
                            self.implementations.append(implementation)
                            self.implementations.sort(key=lambda tup: tup.getCostLoop() )

        return len(self.implementations)

    def generateUtil(self):
        code = ""
        code +="#include \"transpose.h\"\n"
        code +="#include <omp.h>\n"
        code +="#include <fstream>\n"
        code +="#include <float.h>\n"
        code +="#include <stdlib.h>\n"
        code +="#include <stdio.h>\n"
        code +="#include <time.h>\n"
        code +="#include <string>\n"
        if self.architecture == "avx" or self.architecture == "avx512" or self.architecture == "knc":
            code +="#include <immintrin.h>\n"
            code +="#include <xmmintrin.h>\n"
        elif self.architecture == "power":
            code += "#include <builtins.h>\n"
            code += "#include <altivec.h>\n"
        code +="#include <complex.h>\n"
        if self.papi:
            code +="#include <papi.h>\n"
        code +="\n"

        hppCode ="#include <complex.h>\n"
        hppCode +="#include <stdio.h>\n"
        hppCode +="#include <float.h>\n"
        hppCode +="#include <omp.h>\n"
        hppCode +="#include <stdlib.h>\n"
        hppCode +="#include <string>\n"
        if self.papi:
            hppCode +="#include <papi.h>\n"
        hppCode +="\n"
#        hppCode += "void printMatrix2Dcomplex(const %s *A, int *size);"%(self.floatTypeA) 
#        hppCode += "void printMatrix2D(const %s *A, int *size);"%(self.floatTypeA) 
#        code +="void printMatrix2Dcomplex(const %s *A, int *size)"%(self.floatTypeA)
#        code +="{\n"
#        code +="   for(int i=0;i < size[0]; ++i){\n"
#        code +="      for(int j=0;j < size[1]; ++j){\n"
#        code +="         printf(\"(%.2e,%.2e) \", creal(A[i + j * size[0]]), cimag(A[i + j * size[0]]));\n"
#        code +="      }\n"
#        code +="      printf(\"\\n\");\n"
#        code +="   }\n"
#        code +="   printf(\"\\n\");\n"
#        code +="}\n"
#
#        code +="void printMatrix2D(const %s *A, int *size)"%(self.floatTypeA)
#        code +="{\n"
#        code +="   for(int i=0;i < size[0]; ++i){\n"
#        code +="      for(int j=0;j < size[1]; ++j){\n"
#        code +="         printf(\"%.8e \", A[i + j * size[0]]);\n"
#        code +="      }\n"
#        code +="      printf(\"\\n\");\n"
#        code +="   }\n"
#        code +="   printf(\"\\n\");\n"
#        code +="}\n"

        hppCode +="void restoreA(const %s *in, %s*out, int total_size);"%(self.floatTypeA,self.floatTypeA)
        code +="void restoreA(const %s *in, %s*out, int total_size)"%(self.floatTypeA,self.floatTypeA)
        code +="{\n"
        code +="   for(int i=0;i < total_size ; ++i){\n"
        code +="      out[i] = in[i];\n"
        code +="   }\n"
        code +="}\n"

        hppCode +="void restoreB(const %s *in, %s*out, int total_size);"%(self.floatTypeB,self.floatTypeB)
        code +="void restoreB(const %s *in, %s*out, int total_size)"%(self.floatTypeB,self.floatTypeB)
        code +="{\n"
        code +="   for(int i=0;i < total_size ; ++i){\n"
        code +="      out[i] = in[i];\n"
        code +="   }\n"
        code +="}\n"

        hppCode +="int equal(const %s *A, const %s*B, int total_size);"%(self.floatTypeB,self.floatTypeB)
        code +="int equal(const %s *A, const %s*B, int total_size)"%(self.floatTypeB,self.floatTypeB)
        code +="{\n"

        code +="  int error = 0;\n" 
        if( self.floatTypeB.find("complex") != -1 ):
            _floatTypeB = "float"
            if( self.floatTypeB.find("double") != -1 ):
                _floatTypeB = "double"
            code +="   const %s *Atmp = (%s*)A;\n"%(_floatTypeB,_floatTypeB)
            code +="   const %s *Btmp= (%s*)B;\n"%(_floatTypeB,_floatTypeB)
            if self.parallelize != 0:
                code +="   #pragma omp parallel for reduction(+:error) \n"
            code +="   for(int i=0;i < 2*total_size ; ++i){\n"
        else:
            _floatTypeB = self.floatTypeB
            code +="   const %s *Atmp= A;\n"%self.floatTypeB
            code +="   const %s *Btmp= B;\n"%self.floatTypeB
            if self.parallelize != 0:
                code +="   #pragma omp parallel for reduction(+:error) \n"
            code +="   for(int i=0;i < total_size ; ++i){\n"

        code +="      double Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];\n"
        code +="      double Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];\n"
        code +="      double max = (Aabs < Babs) ? Babs : Aabs;\n"
        code +="      double diff = (Aabs - Babs);\n"
        code +="      diff = (diff < 0) ? -diff : diff;\n"
        code +="      if(diff > 0){\n"
        code +="        double relError = (diff / max);\n"
        if( self.floatTypeA.find("float") != -1 or self.floatTypeB.find("float") != -1):
            code +="        if(relError > 4e-5){\n"
        else:
            code +="        if(relError > 1e-12){\n"
        code +="         //printf(\"i: %d relError: %.8e\\n\",i,relError);\n"
        code +="         error += 1;\n"
        code +="      }\n"
        code +="      }\n"
        code +="   }\n"
        #code +="   return error;\n"
        code +="   return (error > 0) ? 0 : 1;\n"
        code +="}\n"

        f = open(self.tmpDirectory + "util.cpp",'w')
        f.write(code)
        f.close()
        f = open(self.tmpDirectory + "util.h",'w')
        f.write(hppCode)
        f.close()

    def printMain(self):
        code = ""
        code +="#include \"transpose.h\"\n"
        code +="#include \"util.h\"\n"
        code +="#include \"measure.h\"\n"
        code +="#include <fstream>\n"
        code +="#include <time.h>\n"
        if self.mpi:
            code +="#include <mpi.h>\n"
        if self.architecture == "avx" or self.architecture == "avx512" or self.architecture == "knc":
            code +="#include <immintrin.h>\n"
            code +="#include <xmmintrin.h>\n"
        elif self.architecture == "power":
            code += "#include <builtins.h>\n"
            code += "#include <altivec.h>\n"
        code +="#include <complex.h>\n"
        code +="\n"

        self.generateUtil()

        code +="\n"
        if self.papi:
            code +="int PapiEventSet;\n"
        code +="int main(int argc, char** argv)\n"
        code +="{\n"
        if self.mpi:
           code +="   MPI_Init(&argc, &argv);\n"
        if self.papi:
           code +="   int retval;\n"
           code +="   PapiEventSet = PAPI_NULL;\n"

           code +="   /* Initialize the PAPI library */\n"
           code +="   retval = PAPI_library_init(PAPI_VER_CURRENT);\n"

           code +="   if (retval != PAPI_VER_CURRENT) {\n"
           code +="      fprintf(stderr, \"PAPI library init error!\\n\");\n"
           code +="      exit(1);\n"
           code +="   }\n"

           code +="   /* Create the Event Set */\n"
           code +="   if (PAPI_create_eventset(&PapiEventSet) != PAPI_OK)\n"
           code +="      fprintf(stderr,\"Error: Papi event not available.\\n\");\n"

           code +="   if (PAPI_add_event(PapiEventSet, PAPI_TLB_DM) != PAPI_OK)\n"
           code +="      fprintf(stderr,\"Error: Papi event not available\\n\");\n"
           code +="   if (PAPI_add_event(PapiEventSet, PAPI_L2_DCM) != PAPI_OK)\n"
           code +="      fprintf(stderr,\"Error: Papi event not available\\n\");\n"
           #code +="   if (PAPI_add_event(PapiEventSet, PAPI_CA_INV) != PAPI_OK)\n"
           #code +="      fprintf(stderr,\"Error: Papi invalidate event not available\\n\");\n"


        code +="   srand(time(NULL));\n"
        code +="\n"
        code +="   double start;\n"
        code +="   int nRepeat = 4;\n"
        code +="   if(argc > 2) nRepeat = atoi(argv[2]);\n"
        code +="   int dim = %d;\n"%self.dim

        line = "   int size[] = {"
        for i in range(self.dim):
            line += str(self.size[i])
            if i != self.dim -1:
                line += ","
        line += "};\n"
        code +=line

        maxSizeA = 1
        if( len(self.lda) == 0):
            for s in self.size:
                maxSizeA *= s
            line = "   int *lda = NULL;\n"
        else:
            for s in self.lda:
                maxSizeA *= s
            line = "   int lda[] = {"
            for i in range(self.dim):
                line += str(self.lda[i])
                if i != self.dim -1:
                    line += ","
            line += "};\n"
        code +=line

        maxSizeB = 1
        if( len(self.ldb) == 0):
            for s in self.size:
                maxSizeB *= s
            line = "   int *ldb = NULL;\n"
        else:
            for s in self.ldb:
                maxSizeB *= s
            line = "   int ldb[] = {"
            for i in range(self.dim):
                line += str(self.ldb[i])
                if i != self.dim -1:
                    line += ","
            line += "};\n"
        code +=line


        maxSize = max(maxSizeA, maxSizeB)
        code +="   int total_size = %d;\n"%(maxSize)
        code +="   int elements_moved = 1;\n"
        code +="   //compute total size\n"
        code +="   for(int i=0;i < dim; ++i){\n"
        code +="      elements_moved *= size[i];\n"
        code +="   }\n"
        code +="\n"
        code +="   double *trash1, *trash2;\n"
        code +="   %s *A, *A_copy;\n"%(self.floatTypeA)
        code +="   %s *B, *B_ref, *B_copy;\n"%(self.floatTypeB)
        code +="   double time;\n"
        if( self.floatTypeA.find("double") != -1 ):
            code +="   const double alpha = %f;\n"%(self.alpha)
        else:
            code +="   const float alpha = %f;\n"%(self.alpha)
        if( self.floatTypeB.find("double") != -1 ):
            code +="   const double beta = %f;\n"%(self.beta)
        else:
            code +="   const float beta = %f;\n"%(self.beta)
        code +="   int largerThanL3 = 1024*1024*100/sizeof(double); \n"
        code +="   int ret = posix_memalign((void**) &trash1, %d, sizeof(double) * largerThanL3);\n"%(self.alignmentRequirement)
        code +="   ret += posix_memalign((void**) &trash2, %d, sizeof(double) * largerThanL3);\n"%(self.alignmentRequirement)
        code +="   ret += posix_memalign((void**) &A, %d, sizeof(%s) * total_size);\n"%(self.alignmentRequirement, self.floatTypeA)
        code +="   ret += posix_memalign((void**) &B_ref, %d, sizeof(%s) * total_size);\n"%(self.alignmentRequirement, self.floatTypeB)
        code +="   ret += posix_memalign((void**) &B_copy, %d, sizeof(%s) * total_size);\n"%(self.alignmentRequirement, self.floatTypeB)
        code +="   ret += posix_memalign((void**) &A_copy, %d, sizeof(%s) * total_size);\n"%(self.alignmentRequirement, self.floatTypeA)
        code +="   ret += posix_memalign((void**) &B, %d, sizeof(%s) * total_size);\n"%(self.alignmentRequirement, self.floatTypeB)
        code +="   if( ret != 0){ printf(\"[TTC] ERROR: posix_memalign failed\\n\"); exit(-1); }\n"
        code +="   const %s *A_const = A;\n"%(self.floatTypeA)
        code +="   const %s *B_copy_const = B_copy;\n"%(self.floatTypeB)
        code +="\n"
        if self.parallelize != 0:
            code +="   #pragma omp parallel for\n"
        code +="   for(int i=0;i < largerThanL3; ++i){\n"
        code +="      trash1[i] = 0;\n"
        code +="      trash2[i] = 0;\n"
        code +="   }\n"
        if self.floatTypeA.find("complex") != -1:
            tmpTypeA = "float"
            if self.floatTypeA.find("double") != -1:
                tmpTypeA = "double"
            tmpTypeB = "float"
            if self.floatTypeB.find("double") != -1:
                tmpTypeB = "double"
            code +="   %s *Atmp = (%s*) A;\n"%(tmpTypeA,tmpTypeA)
            code +="   %s *Btmp = (%s*) B;\n"%(tmpTypeB,tmpTypeB)
            if self.parallelize != 0:
                code +="   #pragma omp parallel for\n"
            code +="   for(int i=0;i < 2*total_size ; ++i){\n"
            code +="      Atmp[i] = (%s)i;\n"%(tmpTypeA)
            code +="      Btmp[i] = (%s)i;\n"%(tmpTypeB)
            code +="   }\n"
            if self.parallelize != 0:
                code +="   #pragma omp parallel for\n"
            code +="   for(int i=0;i < total_size ; ++i){\n"
            code +="      B_ref[i] = B[i];\n"
            code +="      B_copy[i] = B[i];\n"
            code +="      A_copy[i] = A[i];\n"
            code +="   }\n"

        else:
            if self.parallelize != 0:
                code +="   #pragma omp parallel for\n"
            code +="   for(int i=0;i < total_size ; ++i){\n"
            code +="      A[i] = (%s)i;\n"%(self.floatTypeA)
            code +="      B[i] = (%s)i;\n"%(self.floatTypeB)
            code +="      B_ref[i] = B[i];\n"
            code +="      B_copy[i] = B[i];\n"
            code +="      A_copy[i] = A[i];\n"
            code +="   }\n"
        code +="\n"
        if self.mpi:
            code +=  "   int rank, numRanks;\n"
            code +=  "   MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n"
            code +=  "   MPI_Comm_size(MPI_COMM_WORLD, &numRanks);\n"
        code +="   /***************************************************\n"
        code +="   *make sure that all versions yield the same result\n"
        code +="   ***************************************************/\n"
        code +="   double referenceBandwidth = 0;\n"
        if( self.noTest == 0 ):
            if(self.beta != 0 ):
                code +="   %s(A_const, B_ref, alpha, beta, size, lda, ldb);\n"%self.referenceImplementation.getTransposeName()
            else: 
                code +="   %s(A_const, B_ref, alpha, size, lda, ldb);\n"%self.referenceImplementation.getTransposeName() 

        refVersionStr = self.referenceImplementation.getTransposeName()
        #time reference version
        code +="   //time reference version\n"
        code +="   if( argc == 1 || argc >= 2 && std::string(\"" + refVersionStr + "\").compare(argv[1]) == 0){\n"
        code +="      time = FLT_MAX;\n"
        code +="      for(int i = 0; i < nRepeat; i++){\n"
        if( self.noTest == 0 ):
            code +="         if( i < 2 )\n"
            code +="            restoreB(B_copy_const, B, total_size);\n"
        code +="         trashCache(trash1, trash2,largerThanL3);\n"

        if( self.hotB ):
            code +="            restoreB(B_copy_const, B, total_size);\n"
        if( self.hotA ):
            code +="            restoreA(A_const, A_copy, total_size);\n"

        if( self.mpi ):
            code +="         MPI_Barrier(MPI_COMM_WORLD);\n"
        code +="         start = omp_get_wtime();\n"
        if(self.beta != 0 ):
            code +="         %s(A_const, B, alpha, beta, size, lda, ldb);\n"%refVersionStr
        else:
            code +="         %s(A_const, B, alpha, size, lda, ldb);\n"%refVersionStr 
        if( self.mpi ):
            code +="         MPI_Barrier(MPI_COMM_WORLD);\n"
        code +="         double tmpTime = omp_get_wtime() - start;\n"
        code +="         if( tmpTime < time ) time = tmpTime;\n"
        code +="      }\n"
        if self.beta != 0:
            code +="      double bandwidth = ((double)(elements_moved * (sizeof("+self.floatTypeA+") + 2.0 * sizeof("+self.floatTypeB+"))))/(1<<30)/(time);\n"
        else:
            code +="      double bandwidth = ((double)(elements_moved * (sizeof("+self.floatTypeA+") + 1.0 * sizeof("+self.floatTypeB+"))))/(1<<30)/(time);\n"
        code +="      if( time <= 0.0) bandwidth = 100;\n" #if the transpose didn't take enough time too measure it, we just fix the bandwidth to 100 #TODO
        code +="      referenceBandwidth = bandwidth;\n"
        if( self.mpi ):
            code +="      referenceBandwidth *= numRanks;\n"
            code +="      if(rank == 0)\n"
        code +="      printf(\"reference version %s took %%e and achieved %%.2f GB/s \\n\",time, referenceBandwidth );\n"%refVersionStr
        code +="      fflush(stdout);\n"
        code +="   }\n"

        code +="   double maxBandwidth = -1;\n"
        code +="   double maxTop1Bandwidth = -1;\n"
        code +="   double maxTop5Bandwidth = -1;\n"
        code +="   double tmpBandwidth = -1;\n"

        counter = 0

        numImplementations = len(self.implementations)
        numFiles = max(1,(numImplementations + self.minImplementationsPerFile -1)  / self.minImplementationsPerFile)
        if( numFiles > 20 ):
            numFiles = (numImplementations + self.maxImplementationsPerFile -1)  / self.maxImplementationsPerFile
        numSolutionsPerFile = (numImplementations + numFiles - 1) / numFiles

        loopCosts = []
        for impl in self.implementations:
            loopCosts.append(impl.getCostLoop())
        loopCosts = list(set(loopCosts))
        loopCosts.sort()

        #split measurement into several files
        measureHPP = ""
        for i in range(numFiles):
            code += "   tmpBandwidth = measure%d(nRepeat, argc, argv, A_const, A_copy, B, B_copy_const, B_ref, alpha, beta, total_size, elements_moved, largerThanL3, size, trash1, trash2, lda, ldb);\n"%(i)
            code += "   maxBandwidth = (tmpBandwidth < maxBandwidth) ? maxBandwidth : tmpBandwidth;\n"

            tmpCode = "#include \"util.h\"\n"
            tmpCode += "#include \"transpose.h\"\n"
            if self.mpi:
                tmpCode += "#include <mpi.h>\n"
            alphaFloatType = "float"
            if( self.floatTypeA.find("double") != -1 ):
                alphaFloatType = "double"
            betaFloatType = "float"
            if( self.floatTypeB.find("double") != -1 ):
                betaFloatType = "double"
            header = """double measure%d(int nRepeat, int argc, char** argv, const %s *
            A_const, %s * A_copy, %s * B, const %s * B_copy_const, const %s * B_ref, const %s alpha,
            const %s beta, int total_size, int elements_moved,
            int largerThanL3, int *size, double *trash1, double
            *trash2, int* lda, int* ldb)"""%(i,self.floatTypeA,self.floatTypeA,self.floatTypeB,self.floatTypeB,self.floatTypeB,alphaFloatType , betaFloatType)
            if self.papi:
                tmpCode +=  "extern int PapiEventSet;\n"
            tmpCode +=  header + "{\n"
            if self.mpi:
                tmpCode +=  "   int rank, numRanks;\n"
                tmpCode +=  "   MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n"
                tmpCode +=  "   MPI_Comm_size(MPI_COMM_WORLD, &numRanks);\n"
            measureHPP += header + ";\n"
            tmpCode += "   double maxBandwidth = -1;\n"
            tmpCode += "   long long values[3];\n"
            for j in range(i * numSolutionsPerFile, min(numImplementations,(i+1)*numSolutionsPerFile)):
                implementation = self.implementations[j] 
                transposeName = implementation.getTransposeName()
                versionStr = implementation.getVersionName()

                tmpCode +="   if( argc == 1 || argc >= 2 && std::string(\"" + versionStr + "\").compare(argv[1]) == 0){\n"
                tmpCode += "     long long tlb_misses = 0;\n"
                tmpCode += "     long long l2misses = 0;\n"
                tmpCode += "     long long invalidates = 0;\n"
                tmpCode +="      double time = FLT_MAX;\n"
                tmpCode +="      for(int i = 0; i < nRepeat; i++){\n"
                if( self.noTest == 0 ):
                    tmpCode +="         if( i < 2 )\n"
                    tmpCode +="            restoreB(B_copy_const, B, total_size);\n"
                tmpCode +="         trashCache(trash1, trash2,largerThanL3);\n"

                if( self.hotB ):
                    tmpCode +="            restoreB(B_copy_const, B, total_size);\n"
                if( self.hotA ):
                    tmpCode +="            restoreA(A_const, A_copy, total_size);\n"

                if self.mpi:
                    tmpCode +="         MPI_Barrier(MPI_COMM_WORLD);\n"
                tmpCode +="         double start = omp_get_wtime();\n"

                if self.papi:
                    tmpCode +="         /* Start counting */\n"
                    tmpCode +="         if (PAPI_start(PapiEventSet) != PAPI_OK)\n"
                    tmpCode +="            printf(\"Error: papi_start\\n\");\n"
                if(self.beta != 0):
                    tmpCode +="         %s(A_const, B, alpha, beta, size, lda, ldb);\n"%transposeName
                else:
                    tmpCode +="         %s(A_const, B, alpha, size, lda, ldb);\n"%transposeName  
                if self.papi:
                    tmpCode +="         if (PAPI_stop(PapiEventSet, values) != PAPI_OK)\n"
                    tmpCode +="            printf(\"Error: papi_stop\\n\");\n"
                    tmpCode +="         tlb_misses += values[0];\n"
                    #tmpCode +="         l2misses += values[1];\n"
                    #tmpCode +="         invalidates += values[2];\n"
                if self.mpi:
                    tmpCode +="         MPI_Barrier(MPI_COMM_WORLD);\n"
                tmpCode +="         double tmpTime = omp_get_wtime() - start;\n"
                tmpCode +="         if( tmpTime < time ) time = tmpTime;\n"
                if( self.noTest == 0 ):
                    tmpCode +="         if(i == 0 && !equal(B_ref, B, total_size) ){\n"
                    #tmpCode +="           printf(\"B_ref:\\n\");\n"
                    #tmpCode +="           printMatrix2D(B_ref, size);\n"
                    #tmpCode +="           printf(\"B:\\n\");\n"
                    #tmpCode +="           printMatrix2D(B, size);\n"
                    tmpCode +="           printf(\"ERROR version "+versionStr+" doesn't give the same result (line %d)\\n\",__LINE__);\n"
                    tmpCode +="           exit(-1);\n"
                    tmpCode +="         };\n"
                tmpCode +="      }\n"
                if self.beta != 0:
                    tmpCode +="      double bandwidth = ((double)(elements_moved * (sizeof("+self.floatTypeA+") + 2.0 * sizeof("+self.floatTypeB+"))))/(1<<30)/(time);\n"
                else:
                    tmpCode +="      double bandwidth = ((double)(elements_moved * (sizeof("+self.floatTypeA+") + 1.0 * sizeof("+self.floatTypeB+"))))/(1<<30)/(time);\n"
                tmpCode +="      if( time <= 0.0) bandwidth = 100;\n" #if the transpose didn't take enough time too measure it, we just fix the bandwidth to 100 #TODO
                if( self.mpi ):
                    tmpCode +="      bandwidth *= numRanks;\n"
                tmpCode +="      if( bandwidth > maxBandwidth ) maxBandwidth = bandwidth;\n"


                blockingRank = -1
                for rank in range(len(self.blockings)):
                    if self.blockings[rank] == implementation.getBlocking():
                        blockingRank = rank
                        break
                loopRank = loopCosts.index(implementation.getCostLoop())
                #if( self.loopRank.has_key(tuple(implementation.getLoopPerm())) ):
                #    loopRank = self.loopRank[tuple(implementation.getLoopPerm())]

                if( self.mpi ):
                    tmpCode +="      if(rank == 0)\n"
                tmpCode +="      printf(\"variant "+versionStr+" took %%e and achieved %%.2f GB/s (blocking rank: %d) (loop rank: %d) (l2 misses: %%f) (invalidates: %%f)\\n\",time, bandwidth,l2misses/((float)nRepeat),invalidates/((float)nRepeat));\n"%(blockingRank, loopRank)
                tmpCode +="      fflush(stdout);\n"
                tmpCode +="   }\n"
                counter += 1
            tmpCode +="   return maxBandwidth;\n"
            tmpCode +="}\n"
            f = open(self.tmpDirectory + "measure%d.cpp"%i,'w')
            f.write(tmpCode)
            f.close()
            f = open(self.tmpDirectory + "measure.h",'w')
            f.write(measureHPP)
            f.close()

        code +="   /***************************************************/\n"

        if( self.mpi ):
            code +="      if(rank == 0){\n"
        code +="   printf(\"Maximal bandwidth: %f\\n\", maxBandwidth);\n"
        code +="   printf(\"Speedup over reference: %f\\n\", maxBandwidth / referenceBandwidth );\n"
        code +="   printf(\"Top-1 speedup: %.2f\\n\", maxTop1Bandwidth/maxBandwidth);\n"
        code +="   printf(\"Top-5 speedup: %.2f\\n\", maxTop5Bandwidth/maxBandwidth);\n"
        code +="   printf(\"SUCCESS!\\n\");\n"
        if( self.mpi ):
            code +="      }\n"
        code +="   free(A); free(B);\n"
        code +="   free(A_copy); free(B_copy);\n"
        code +="   free(B_ref);\n"
        code +="   free(trash1);\n"
        code +="   free(trash2);\n"
        if self.mpi:
           code +="   MPI_Finalize();\n"
        code +="   return 0;\n"
        code +="}\n"
        f = open(self.tmpDirectory + "main.cpp",'w')
        f.write(code)
        f.close()

    def getScalarFraction(self,blocking):
        remainderA = (self.size[0] % blocking[0])
        fractionA = remainderA / self.size[0]
        remainderB = (self.size[self.perm[0]] % blocking[1])
        fractionB = remainderB / self.size[self.perm[0]]
        return max(fractionA, fractionB)

    def getTranspositionMicroKernel(self):
        # we choose the precision based on the input tensor A
        availableBlocking = []

        kernelName = self.floatTypeA
        if( self.floatTypeA == "double complex" ):
            kernelName = "doubleComplex"
        elif( self.floatTypeA == "float complex" ):
            kernelName = "complex"

        found = 0
        for filename in os.listdir("./micro-kernels"):
            if( filename.find( self.architecture ) != -1 ):
                if( filename.find( kernelName +"_"+self.architecture+ ".kernel") != -1 ):
                    blocking = (int(filename.split("_")[1].split("x")[0]),
                                int(filename.split("_")[1].split("x")[1]))
                    f = open("./micro-kernels/"+filename,'r')
                    code = self.indent +"//%dx%d transpose micro kernel\n"%(blocking[0],blocking[1])
                    code += f.read() + "\n"
                    f.close()

                    availableBlocking.append( (blocking, code) )
                    found += 1

        if( found <= 0 ):
            print "ERROR: no suitable kernels found."
            exit(-1)

        #determine which micro-blocking to use
        availableBlocking = sorted(availableBlocking, key = lambda tup : tup[0][0], reverse=True) # sort blockings from large to small
        for (blocking, code) in availableBlocking:
           scalarFraction = self.getScalarFraction(blocking)
           if( scalarFraction >= 0.33 ):
              continue
           else:
              return (blocking,code)

        return availableBlocking[0]

    def getLoadKernel(self, A, lda, floatType, mixedPrecision, offset, define):
        code = self.indent +"//Load %s\n"%A
        maxRange = self.registerSizeBits / 8 / self.floatSizeA

        if( mixedPrecision ):
            if( self.architecture != "avx" ):
                print FAIL + "Error: mixed precision is not yet supported for this architecture.\n" + ENDC
                exit(-1)

        vectorType = "__m256"
        cast = ""
        if( self.architecture == "avx" or self.architecture == "avx512" or self.architecture == "knc"): #-------------- avx ---------------------
            if( floatType == "float" or floatType == "float complex"):
                if( floatType == "float complex" ):
                    cast = "(const float*)"
                if( mixedPrecision and self.architecture == "avx" ):
                    vectorType = "__m128"
                    if( self.aligned ):
                        functionName = "_mm_load_ps"
                    else:
                        functionName = "_mm_loadu_ps"
                else:
                    vectorType = "__m%d"%self.registerSizeBits
                    if( self.aligned ):
                       if( self.registerSizeBits == 128 ):
                          functionName = "_mm_load_ps"
                       else:
                          functionName = "_mm%d_load_ps"%self.registerSizeBits
                    else:
                       if( self.registerSizeBits == 128 ):
                          functionName = "_mm_loadu_ps"
                       else:
                          functionName = "_mm%d_loadu_ps"%self.registerSizeBits
            elif( floatType == "double" or floatType == "double complex"):
                if( floatType == "double complex" ):
                    cast = "(const double*)"
                if( mixedPrecision and self.architecture == "avx" ):
                    vectorType = "__m256d"
                    if( self.aligned ):
                        functionName = "_mm256_load_pd"
                    else:
                        functionName = "_mm256_loadu_pd"
                else:
                    vectorType = "__m%dd"%self.registerSizeBits
                    if( self.aligned ):
                       if( self.registerSizeBits == 128 ):
                          functionName = "_mm_load_pd"
                       else:
                          functionName = "_mm%d_load_pd"%self.registerSizeBits
                    else:
                       if( self.registerSizeBits == 128 ):
                          functionName = "_mm_loadu_pd"
                       else:
                          functionName = "_mm%d_loadu_pd"%self.registerSizeBits
            else:
                print FAIL + "Error: unknown datatype.\n" + ENDC
                exit(-1)

        elif(self.architecture == "power"):
            vectorType = "vector4double"
        else:
            print FAIL + "Error: architecture unknown.\n" + ENDC
            exit(-1)

        if( define == 0 ):
            vectorType = ""

        for i in range(maxRange):
            if( self.aligned ):
                if(self.architecture == "power"):
                    code += self.indent + "%s row%s%d = vec_lda(0,const_cast<float*>(%s+%d+%d*%s));\n"%(vectorType,A,i,A,offset,i,lda)
                else:
                    code += self.indent + "%s row%s%d = %s(%s(%s + %d +%d*%s));\n"%(vectorType,A,i,functionName,cast,A,offset,i,lda)
            else:
                if(self.architecture == "power"):
                    print "non-aligned loads are not yet supported for Power ."
                    exit(-1)
                else:
                    code += self.indent + "%s row%s%d = %s(%s(%s+%d+%d*%s));\n"%(vectorType,A,i,functionName,cast,A,offset,i,lda)

        return code + "\n"

    def getStoreKernel(self, reg, offset):
        code = self.indent +"//Store B\n"
        maxRange = self.registerSizeBits / 8 / self.floatSizeA
        cast = ""
        if( self.architecture == "avx" or self.architecture == "avx512" or self.architecture == "knc" ):
            post = "ps"
            if( self.floatTypeB.find("double") != -1 ):
                post = "pd"

            if( self.aligned ):
                if( self.floatSizeB < self.floatSizeA ): # mixed precision 
                    functionName = "_mm_store_%s"%(post)
                else:
                   if( self.registerSizeBits == 128 ):
                      functionName = "_mm_store_%s"%(post)
                   else:
                      functionName = "_mm%d_store_%s"%(self.registerSizeBits,post)
            else:
                if( self.floatSizeB < self.floatSizeA ): # mixed precision 
                    functionName = "_mm_storeu_%s"%(post)
                else:
                    if( self.registerSizeBits == 128 ):
                       functionName = "_mm_storeu_%s"%(post)
                    else:
                       functionName = "_mm%d_storeu_%s"%(self.registerSizeBits,post)

            if( self.floatTypeB == "float complex" ):
                cast = "(float*)"
            elif( self.floatTypeB == "double complex" ):
                cast = "(double*)"

        elif(self.architecture == "power"):
            functionName = "vec_sta"


        for i in range(maxRange):
            if( self.aligned ):
                if(self.architecture == "power"):
                    code += self.indent + "%s(%s, 0 ,B + %d + %i * ldb);\n"%(functionName, reg.replace("#",str(i)),offset,i)
                else:
                    code += self.indent + "%s(%s(B + %d + %i * ldb), %s);\n"%(functionName,cast,offset, i,reg.replace("#",str(i)))
            else:
                if(self.architecture == "power"):
                    print "non-aligned stores are not yet supported for Power ."
                    exit(-1)
                else:
                    code += self.indent + "%s(%s(B + %d + %i * ldb), %s);\n"%(functionName,cast,offset, i,reg.replace("#",str(i)))
        return code

    def getScaleKernel(self, A, alpha):
        code = self.indent +"//Scale %s\n"%A
        maxRange = self.registerSizeBits / 8 / self.floatSizeA
        if( self.floatTypeA == "float" or self.floatTypeA == "float complex"): 
           if( self.registerSizeBits == 128 ):
              functionName = "_mm_mul_ps"
           else:
              functionName = "_mm%d_mul_ps"%self.registerSizeBits
        if( self.floatTypeA == "double" or self.floatTypeA == "double complex"):
           if( self.registerSizeBits == 128 ):
              functionName = "_mm_mul_pd"
           else:
              functionName = "_mm%d_mul_pd"%self.registerSizeBits

        for i in range(maxRange):
            if(self.architecture == "power"):
                code += self.indent + "row%s%d = vec_mul(row%s%d, %s);\n"%(A,i,A,i,alpha)
            else:
                code += self.indent + "row%s%d = %s(row%s%d, %s);\n"%(A,i,functionName,A,i,alpha)

        return code + "\n"
   
    #d = a * b + c
    def getFmaKernel(self, a, b, c, d):
        code = ""
        if(self.architecture == "power"):
            code += self.indent + "%s = vec_madd( %s, %s, %s);\n"%(d,a,b,c)
        else:
            if( self.floatTypeB.find("float") != -1):
                if(  self.architecture == "avx512" or self.architecture == "knc" ): 
                    code += self.indent + "%s = _mm512_fmadd_ps( %s, %s, %s);\n"%(d,a,b,c)
                else:
                    if( self.floatTypeA.find("double") != -1  and self.floatTypeB.find("float") != -1): #mixed precision
                        code += self.indent + "%s = _mm_add_ps( _mm_mul_ps(%s, %s), _mm256_cvtpd_ps(%s));\n"%(d,a,b,c)
                    else:
                        if( self.registerSizeBits == 128 ):
                            code += self.indent + "%s = _mm_add_ps( _mm_mul_ps(%s, %s), %s);\n"%(d,a,b,c)
                        else:
                            code += self.indent + "%s = _mm256_add_ps( _mm256_mul_ps(%s, %s), %s);\n"%(d,a,b,c)
            if( self.floatTypeB.find("double") != -1): 
                if(  self.architecture == "avx512" or self.architecture == "knc" ): 
                    code += self.indent + "%s = _mm512_fmadd_pd( %s, %s, %s);\n"%(d,a,b,c)
                else:
                    if( self.floatTypeA.find("float") != -1): #mixed precision
                        code += self.indent + "%s = _mm256_add_pd( _mm256_mul_pd(%s, %s), _mm256_cvtps_pd((%s)));\n"%(d,a,b,c)
                    else:
                        if( self.registerSizeBits == 128 ):
                            code += self.indent + "%s = _mm_add_pd( _mm_mul_pd(%s, %s), %s);\n"%(d,a,b,c)
                        else:
                            code += self.indent + "%s = _mm256_add_pd( _mm256_mul_pd(%s, %s), %s);\n"%(d,a,b,c)
        return code

    def getBroadcastVariables(self, withType):
        code = " "
        if(self.perm[0] != 0 and self.scalar == 0):

            alphaFloatType = "__m%d"%self.registerSizeBits
            if(self.floatTypeA =="double" or self.floatTypeA =="double complex"):
                alphaFloatType = "__m%dd"%self.registerSizeBits
            if(self.architecture == "power"):
                alphaFloatType = "vector4double"

            betaFloatType = "__m%d"%self.registerSizeBits
            if( (self.floatTypeB.find("float") != -1) and (self.floatTypeA.find("double") != -1)):
                betaFloatType = "__m%d"%(self.registerSizeBits/2)

            if(self.floatTypeB =="double" or self.floatTypeB =="double complex"):
                betaFloatType = "__m%dd"%self.registerSizeBits
            if(self.architecture == "power"):
                betaFloatType = "vector4double"

            if(withType==1):
                code += " ,const %s &reg_alpha"%alphaFloatType 
            else:
                code += " , reg_alpha"

	    if(self.beta !=0):
                if(withType==1):
                    code += " ,const %s &reg_beta"%betaFloatType 
                else:
                    code += " , reg_beta"
        else:
            alphaFloatType = "float"
            if( self.floatTypeA.find("double") != -1 ):
                alphaFloatType = "double"
            betaFloatType = "float"
            if( self.floatTypeB.find("double") != -1 ):
                betaFloatType = "double"
            if(withType==1):
                code += " ,const %s alpha"%(alphaFloatType)
                if(self.beta !=0 ):
                    code += " ,const %s beta"%(betaFloatType) 
            else:
                code += " ,alpha"
                if(self.beta !=0 ):
                    code += " ,beta"    
                
        return code 

    def getMicroKernelHeader(self,blocking, prefetchDistance = 0, staticAndInline = 0):
        code = "//B_ji = alpha * A_ij + beta * B_ji\n"
        transposeMicroKernelname = "%sTranspose%dx%d"%(ttc_util.getFloatPrefix(self.floatTypeA, self.floatTypeB),blocking[0], blocking[1])
        if( self.perm[0] == 0):
            transposeMicroKernelname += "_0"

        if( ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
            transposeMicroKernelname += "_streamingstore"

        if( self.beta == 0 ):
            transposeMicroKernelname += "_bz"

        if( prefetchDistance > 0):
            transposeMicroKernelname += "_prefetch_%d"%prefetchDistance

        #if( staticAndInline ):
        #    transposeMicroKernelname += "_"
        #    for i in self.perm:
        #        transposeMicroKernelname += str(i)
  
        #    transposeMicroKernelname +="_"
        #    for idx in range(len(self.size)):
        #         transposeMicroKernelname += "%d"%(self.size[idx])
        #         if(idx != len(self.size)-1):
        #             transposeMicroKernelname +="x"

        static = ""
        if staticAndInline :
            static = "static INLINE "
        if( prefetchDistance > 0):
            return transposeMicroKernelname, code +static+"void %s(const %s* __restrict__ A, const int lda, %s* __restrict__ B, const int ldb, const %s* __restrict__ Anext0, %s* __restrict__ Bnext0, const %s* __restrict__ Anext1, %s* __restrict__ Bnext1%s)\n{\n"""%(transposeMicroKernelname, self.floatTypeA,self.floatTypeB, self.floatTypeA,self.floatTypeB, self.floatTypeA,self.floatTypeB,self.getBroadcastVariables(1))
        else:
            if( self.perm[0] != 0):
                return transposeMicroKernelname, code +static+"void %s(const %s* __restrict__ A, const int lda, %s* __restrict__ B, const int ldb%s)\n{\n"%(transposeMicroKernelname, self.floatTypeA, self.floatTypeB,self.getBroadcastVariables(1))
            else:
                if( staticAndInline ):
                    return transposeMicroKernelname, code +"template<int size0>\nvoid %s(const %s* __restrict__ A, int lda1, const int lda, %s* __restrict__ B, const int ldb1, const int ldb%s)\n{\n"%(transposeMicroKernelname, self.floatTypeA, self.floatTypeB,self.getBroadcastVariables(1))
                else:
                    return transposeMicroKernelname, code +static+"void %s(const %s* __restrict__ A, int lda1, const int lda, %s* __restrict__ B, const int ldb1, const int ldb%s)\n{\n"%(transposeMicroKernelname, self.floatTypeA, self.floatTypeB,self.getBroadcastVariables(1))


    def getUpdateAndStore(self):

        numIterations = 1
        if( self.floatTypeA.find("float") != -1 and self.floatTypeB.find("double") != -1):
            numIterations = 2

        code = ""
        for iteration in range(numIterations):
            offset = (self.registerSizeBits / 8 / self.floatSizeB) * iteration

            if( self.beta != 0 ):
                loadKernelB = self.getLoadKernel("B","ldb", self.floatTypeB, self.floatTypeA != self.floatTypeB, offset, iteration == 0)

            if( iteration == 0 ):
                code += self.getScaleKernel("A", "reg_alpha")
            if(self.beta != 0 ):
                code += loadKernelB
                for i in range(self.microBlocking[0][0]):
                    if( self.floatTypeA.find("float") != -1 and self.floatTypeB.find("double") != -1): #mixed precision
                        if( iteration == 0):
                            code += self.getFmaKernel("rowB%d"%i, "reg_beta", "_mm256_castps256_ps128(rowA%d)"%i, "rowB%d"%i)
                        else:
                            code += self.getFmaKernel("rowB%d"%i, "reg_beta", "_mm256_extractf128_ps(rowA%d, 0x1)"%i, "rowB%d"%i)
                    else:
                        code += self.getFmaKernel("rowB%d"%i, "reg_beta", "rowA%d"%i, "rowB%d"%i)
                code += self.getStoreKernel("rowB#", offset)
            else:
                if( self.floatTypeA.find("float") != -1 and self.floatTypeB.find("double") != -1): #mixed precision
                    if( iteration == 0):
                        code += self.getStoreKernel("_mm256_cvtps_pd(_mm256_castps256_ps128(rowA#))", offset)
                    else:
                        code += self.getStoreKernel("_mm256_cvtps_pd(_mm256_extractf128_ps(rowA#, 0x1))", offset)
                elif( self.floatTypeA.find("double") != -1 and self.floatTypeB.find("float") != -1): #mixed precision
                    code += self.getStoreKernel("_mm256_cvtpd_ps(rowA#)", offset)
                else:
                    code += self.getStoreKernel("rowA#", offset)

        return code

    def getPrefetchCode(self, ii, jj, numBlocksI, numBlocksJ, prefetchDistance, opt):

       blockA = self.microBlocking[0][0]
       blockB = self.microBlocking[0][1]

       numBlocksTotal = numBlocksI * numBlocksJ
       blockId = ii * numBlocksJ + jj + prefetchDistance
       tile = 0
       if( blockId >= numBlocksTotal ):
          tile = 1
          blockId = blockId % numBlocksTotal 
       
       iPrefetch = blockId / numBlocksJ
       jPrefetch = blockId % numBlocksJ


       if ( iPrefetch == 0):
          if( jPrefetch == 0):
              offsetA = "Anext%d"%tile
              offsetB = "Bnext%d"%tile
          else:
              offsetA = "Anext%d"%(tile)
              offsetB = "Bnext%d + %d"%(tile,jPrefetch * blockB)
       else:
          if( jPrefetch == 0):
              offsetA = "Anext%d + %d"%(tile,iPrefetch * blockA)
              offsetB = "Bnext%d"%(tile)
          else:
              offsetA = "Anext%d + %d"%(tile,iPrefetch * blockA)
              offsetB = "Bnext%d + %d"%(tile,jPrefetch * blockB)

       numElementsPerCacheLineA = self.cacheLineSize / self.floatSizeA
       numElementsPerCacheLineB = self.cacheLineSize / self.floatSizeB

       code = ""
       if( (iPrefetch * blockA) % numElementsPerCacheLineA == 0 ): #we only prefech once per cache-line
          code += self.indent + "//prefetch A\n"
          for l in range(blockA):
            if( self.architecture == "avx" or self.architecture == "knc" or self.architecture == "avx512" ):
                code += self.indent + "_mm_prefetch((char*)(%s + %d * lda), _MM_HINT_T2);\n"%(offsetA,l + jPrefetch * blockA)
            elif( self.architecture == "power" ):
                code += self.indent + "__prefetch_by_load((const void*)(%s + %d * lda));\n"%(offsetA,l + jPrefetch * blockA)
            else:
                print "ERROR: wrong architecture!"
                exit(-1)

       if( opt != "streamingstore" ):
          if( (jPrefetch * blockB) % numElementsPerCacheLineB == 0 ): #we only prefech once per cache-line
              code += self.indent + "//prefetch B\n"
              for l in range(blockB):
                if( self.architecture == "avx" or self.architecture == "knc" or self.architecture == "avx512" ):
                  code += self.indent + "_mm_prefetch((char*)(%s + %d * ldb), _MM_HINT_T2);\n"%(offsetB,l + iPrefetch * blockB)
                elif( self.architecture == "power" ):
                    code += self.indent + "__prefetch_by_load((const void*)(%s + %d * lda));\n"%(offsetA,l + jPrefetch * blockA)
                else:
                    print "ERROR: wrong architecture!"
                    exit(-1)

       return code


    def generateTranspositionKernel(self, blockings, prefetchDistances, staticAndInline=0, optimizations = []):
        # This function generates the transpose.cpp file (_not_ the transpose%d.cpp files)
        #
        # staticAndInline this is _only_ set if the final/fastest version will be dumped to file

        loadKernelA = self.getLoadKernel("A","lda", self.floatTypeA, 0, 0, 1)

        retHPP = ""
        ret = ""

        #generate DxD micro kernel
        for opt in optimizations:
            if( self.perm[0] != 0 ):
                transposeMicroKernelname, tmpCode = self.getMicroKernelHeader(self.microBlocking[0], 0, staticAndInline)   
                code = ""
                if( staticAndInline ):
                    code += "#ifndef _TTC_%s\n"%transposeMicroKernelname.upper()
                    code += "#define _TTC_%s\n"%transposeMicroKernelname.upper()
                code += tmpCode
                retHPP += tmpCode.split("\n")[1]+";\n"
                if( self.scalar != 0 ):
                    code += "  for(int i=0; i < %d; i++)\n"%self.microBlocking[0][0]
                    code += "     for(int j=0; j < %d; j++)\n"%self.microBlocking[0][1]
                    if(self.beta != 0): 
                        code += "        B[j + i * ldb] = alpha*A[i + j * lda] + beta*B[j + i * ldb];\n"
                    else:
                        code += "        B[j + i * ldb] = alpha*A[i + j * lda];\n"
                else:
                    code += loadKernelA
                    code += self.microBlocking[1]
                    code += self.getUpdateAndStore()
                code += "}\n"
                if( staticAndInline ):
                    code += "#endif\n"
                ret += code

                blockA = self.microBlocking[0][0]
                blockB = self.microBlocking[0][1]
            else:
                blockA = 1
                blockB = 1

            if( blockA != blockB ):
                print "Error: non-square micro-kernels are not supported yet."
                exit(-1)

            if( self.perm[0] != 0):
                for prefetchDistance in prefetchDistances:
                    # generate arbitrary blockings based on the DxD micro kernel
                    for blocking in blockings:
                        if( opt == "streamingstore" and (not ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores)) ):
                            continue #skip this blocking if necessary

                        if ( blocking[0] % blockA == 0 and blocking[1] % blockB == 0 and
                                (blocking[0] / blockA > 1 or blocking[1] / blockB > 1 or
                                    prefetchDistance > 0)):

                            transposeMicroKernelname, tmpCode = self.getMicroKernelHeader(blocking, prefetchDistance, staticAndInline)
                            code = ""
                            if( staticAndInline ):
                                code += "#ifndef _TTC_%s\n"%transposeMicroKernelname.upper()
                                code += "#define _TTC_%s\n"%transposeMicroKernelname.upper()
                            code += tmpCode
                            retHPP += tmpCode.split("\n")[1]+";\n"
                            #replicate the micro-transpose to build the bigger transpose
                            numBlocksA = blocking[0] / blockA
                            numBlocksB = blocking[1] / blockB
                            if( opt == "streamingstore"):
                                code += self.indent + "%s B_buffer[%d * %d] __attribute__((aligned(%d)));\n"%(self.floatTypeB, blocking[0], blocking[1],self.cacheLineSize)
                            for i in range(numBlocksA):
                                for j in range(numBlocksB):
            
                                    offsetA = ""
                                    offsetB = ""
                                    if ( i == 0):
                                        if( j == 0):
                                            offsetA = ""
                                            offsetB = ""
                                        else:
                                            offsetA = " + %d * lda"%(j * blockA)
                                            offsetB = " + %d"%(j * blockB)
                                    else:
                                        if( j == 0):
                                            offsetA =" + %d"%(i * blockA)
                                            offsetB =" + %d * ldb"%(i * blockB)
                                        else:
                                            offsetA = " + %d + %d * lda"%(i * blockA,j * blockA)
                                            offsetB = " + %d + %d * ldb"%(j * blockB,i * blockB)

                                    #prefetch next block
                                    if( prefetchDistance > 0 ):
                                        #Citation form the Intel Optimization Manual:
                                        #"It may seem convenient to cluster all of PREFETCH instructions at the beginning of a loop
                                        #body or before a loop, but this can lead to severe performance degradation. In order
                                        #to achieve the best possible performance, PREFETCH instructions must be interspersed
                                        #with other computational instructions in the instruction sequence rather than
                                        #clustered together"
                                        code += self.getPrefetchCode(i,j,numBlocksA,
                                                numBlocksB, prefetchDistance, opt)
                                    transposeName = "%sTranspose%dx%d"%(ttc_util.getFloatPrefix(self.floatTypeA, self.floatTypeB),blockA, blockB)
                                    if( self.perm[0] == 0):
                                        transposeName += "_0"
                                    if( ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
                                        transposeName += "_streamingstore"
                                    if( self.beta == 0 ):
                                        transposeName += "_bz"

                                    code += self.indent + "//invoke micro-transpose\n"
                                    if( opt == "streamingstore"):
                                        code += self.indent + "%s(A%s, lda, B_buffer%s, %d%s);\n\n"%(transposeName, offsetA, offsetB.replace("ldb","%d"%blocking[1]),blocking[1],self.getBroadcastVariables(0))
                                    else:
                                        code += self.indent + "%s(A%s, lda, B%s, ldb%s);\n\n"%(transposeName, offsetA, offsetB,self.getBroadcastVariables(0))

                            if( opt == "streamingstore"):
                                elementsPerRegister = self.registerSizeBits / 8 / self.floatSizeB
                                code += self.indent + "// write buffer to main-memory via non-temporal stores\n"
                                code += self.indent + "for( int i = 0; i < %d; i++){\n"%(blocking[0])
                                if( not ttc_util.streamingStoresApplicable(self.ldb, self.size, self.perm, self.beta, self.cacheLineSize, self.floatSizeB, self.streamingStores) ):
                                    print "ERROR (internal): blockB is not a multiple of the cacheline size"
                                    exit(-1)
                                for j in range((blocking[1] / elementsPerRegister) ): #store one cacheline at a time
                                    cast = ""
                                    if( self.floatTypeB == "float complex" ):
                                        cast = "(float*)"
                                    elif( self.floatTypeB == "double complex" ):
                                        cast = "(double*)"
                                    post = "ps"
                                    if( self.floatTypeB.find("double") != -1 ):
                                        post = "pd"
                                    if( self.registerSizeBits == 128 ):
                                        functionNameStream = "_mm_stream_%s"%post
                                        functionNameLoad = "_mm_load_%s"%post
                                    else:
                                        functionNameStream = "_mm%d_stream_%s"%(self.registerSizeBits,post)
                                        functionNameLoad = "_mm%d_load_%s"%(self.registerSizeBits,post)
                                    code += self.indent + self.indent + "%s(%s(B + i * ldb + %d), %s(%s(B_buffer + i * %d + %d)));\n"%(functionNameStream, cast, j *
                                            elementsPerRegister, functionNameLoad, cast, blocking[1], j * elementsPerRegister)
                                code += self.indent + "}\n"


                            code += "}\n"
                            if( staticAndInline ):
                                code += "#endif\n"
                            ret += code
            else: # perm[0] == 0
                tmpBlockings = copy.deepcopy(sorted(blockings)) #it's important to sort the
                                                #blockings in an ascending order because all blockings will
                                                #use the 1x1 as a building block. This is done to trick the
                                                #compiler into issuing vmovntps, when needed
                if( not (tmpBlockings[0][0] == 1 and tmpBlockings[0][1] == 1) ): # (1,1) needs to be present for every blocking
                    tmpBlockings =  [(1,1)] + tmpBlockings
                for blocking in tmpBlockings:
                    transposeMicroKernelname, tmpCode = self.getMicroKernelHeader(blocking, 0, staticAndInline)
                    code = ""
                    if( staticAndInline ):
                        code += "#ifndef _TTC_%s\n"%transposeMicroKernelname.upper()
                        code += "#define _TTC_%s\n"%transposeMicroKernelname.upper()
                    code += tmpCode
                    retHPP += tmpCode.split("\n")[1]+";\n"
                    indent = self.indent
                    alphaFloatType = "float"
                    if( self.floatTypeA.find("double") != -1 ):
                        alphaFloatType = "double"
                    betaFloatType = "float"
                    if( self.floatTypeB.find("double") != -1 ):
                        alphaFloatType = "double"
                    offsetB = ""
                    offsetA = ""
                    if( blocking[0] > 1 ):
                        code += indent + "for(int ia = 0; ia < %d; ia++)\n"%(blocking[0])
                        offsetB += " + ia * ldb"
                        offsetA += " + ia * lda1"
                        indent += self.indent
                    if( blocking[1] > 1 ):
                        code += indent + "for(int ib = 0; ib < %d; ib++)\n"%(blocking[1])
                        offsetB += " + ib * ldb1"
                        offsetA += " + ib * lda"
                        indent += self.indent


                    if( (blocking[0] == 1 and blocking[1] == 1) or opt != "streamingstore"):
                        if self.architecture != "power":
                            if( opt == "streamingstore" ):
                                code += indent + "#pragma vector nontemporal\n"
                            code += indent + "#pragma omp simd\n"

                        if( staticAndInline ):
                            code += indent + "for(int i0 = 0; i0 < size0; i0++)\n"
                        else:
                            code += indent + "for(int i0 = 0; i0 < %d; i0++)\n"%(self.size[0])
                        updateStr = ""
                        outStr = "B[i0%s]"%offsetB
                        if( len(self.size) == 1):
                            inStr = "A[i0]"
                        else:
                            inStr = "A[i0%s]"%offsetA
                        if( self.beta == 0.0 ):
                            updateStr +=  "%s%s = alpha * %s;\n"%(indent + self.indent, outStr, inStr)
                        else:
                            updateStr +=  "%s%s = alpha * %s + beta * %s;\n"%(indent + self.indent, outStr, inStr, outStr)
                        code += updateStr
                    else:
                        streamStr = ""
                        if( opt == "streamingstore" ):
                            streamStr = "_streamingstore"
                        betaStr = ""
                        if( self.beta == 0 ):
                            betaStr = "_bz"
                        if( staticAndInline ):
                            code += indent + "%sTranspose1x1_0%s%s<size0>(A%s, lda1, lda, B%s, ldb1, ldb, alpha);\n"%(ttc_util.getFloatPrefix(self.floatTypeA, self.floatTypeB),streamStr, betaStr, offsetA, offsetB)
                        else:
                            code += indent + "%sTranspose1x1_0%s%s(A%s, lda1, lda, B%s, ldb1, ldb, alpha);\n"%(ttc_util.getFloatPrefix(self.floatTypeA, self.floatTypeB), streamStr, betaStr, offsetA, offsetB)
                    code += "}\n"
                    if( staticAndInline ):
                        code += "#endif\n"
                    ret += code


        return (ret,retHPP)

    #only used in the case of perm[0] == 0
    def getUpdateString(self,indent):
        outStr = "B[i0 + ib * %d + ia * ldb]"%self.size[0]

        inStr = "A[i0 + ia * %d + ib * lda]"%self.size[0]

        ret = ""

        if(self.beta != 0):
            ret +=  "%s%s = alpha*%s + beta*%s;\n"%(indent + self.indent, outStr, inStr,outStr)
        else:
            ret +=  "%s%s = alpha*%s;\n"%(indent + self.indent, outStr, inStr)
  
        return ret



    def getTrashCache(self):
        cppCode = "void trashCache(double *A, double *B, int n)\n"
        cppCode += "{\n"
        if self.parallelize != 0:
            cppCode += "   #pragma omp parallel for\n"
        cppCode += "   for(int i = 0; i < n; i++)\n"
        cppCode += "      A[i] += 0.999 * B[i];\n"
        cppCode += "}\n"

        return cppCode


                        
    def generateImplementations(self):
        #generate CPP and HPP files
        sortedImplementations = copy.deepcopy(self.implementations)
        sortedImplementations.sort(key=lambda x: x.getBlocking()) #sort according to the
                                                                  #blocking, this is done to reduce the overhead during compilation
        numImplementations = len(self.implementations)
        numFiles = max((numImplementations + self.minImplementationsPerFile -1)  / self.minImplementationsPerFile, 1)
        if( numFiles > 20 ):
            numFiles = (numImplementations + self.maxImplementationsPerFile -1)  / self.maxImplementationsPerFile
        numSolutionsPerFile = (numImplementations + numFiles - 1) / numFiles

        cppCode = ""
        if self.architecture == "avx" or self.architecture == "knc" or self.architecture == "avx512":
            cppCode += "#include <xmmintrin.h>\n"
            cppCode += "#include <immintrin.h>\n"
        elif self.architecture == "power":
            cppCode += "#include <builtins.h>\n"
            cppCode += "#include <altivec.h>\n"

        cppCode += "#include <complex.h>\n"
        cppCode += "#if defined(__ICC) || defined(__INTEL_COMPILER)\n"
        cppCode += "#define INLINE __forceinline\n"
        cppCode += "#else\n"
        cppCode += "#define INLINE __attribute__((always_inline))\n"
        cppCode += "#endif\n\n"
        hppCode = ""
        if self.architecture == "avx" or self.architecture == "knc" or self.architecture == "avx512":
            hppCode += "#include <xmmintrin.h>\n"
            hppCode += "#include <immintrin.h>\n"
        elif self.architecture == "power":
            hppCode += "#include <builtins.h>\n"
            hppCode += "#include <altivec.h>\n"
        hppCode += "#include<complex.h>\n"
        tmpPrefetchDistances = list(self.prefetchDistances)
        #we need prefetch distance 0 for the remainder while-loop
        tmpPrefetchDistances.append(0)
        tmpPrefetchDistances = set( tmpPrefetchDistances )

        (retCpp, retHpp) = self.generateTranspositionKernel(self.blockings, tmpPrefetchDistances, 0, self.getAppropriateOptimizations() )
        hppCode +=  retHpp
        hppCode += "void trashCache(double *A, double *B, int n);\n"
        hppCode += self.referenceImplementation.getHeader()
        for implementation in self.implementations:
            hppCode += implementation.getHeader()
        f = open(self.tmpDirectory+"transpose.h",'w')
        f.write(hppCode)
        f.close()

        cppCode +=  retCpp
        f = open(self.tmpDirectory+"transpose.cpp",'w')
        f.write(cppCode)
        f.close()

        self.generateOffsetFile(self.tmpDirectory)
        implementationCounter = 1 #include reference implementation
        for i in range(numFiles):
            cppCode = ""
            cppCode += "#include <stdio.h>\n"
            cppCode += "#include <queue>\n"
            cppCode += "#include <omp.h>\n"
            cppCode += "#include <complex.h>\n"
            cppCode += "#include \"transpose.h\"\n"
            cppCode += "#include \"ttc_offset.h\"\n"

            if( i == 0):
                cppCode += self.getTrashCache()
                cppCode += self.referenceImplementation.getImplementation(self.parallelize)


            cppImplementations = ""
            for j in range(i * numSolutionsPerFile, min(numImplementations,(i+1)*numSolutionsPerFile)):
                implementation = sortedImplementations[j]
                implementationCounter += 1
                cppImplementations += implementation.getImplementation(self.parallelize)

            cppCode += cppImplementations

            f = open(self.tmpDirectory+"transpose%d.cpp"%i,'w')
            f.write(cppCode)
            f.close()

        if(implementationCounter-1 != numImplementations ):
            print FAIL+"ERROR: not all implementations dumped to file"+ENDC
            exit(-1)
            

