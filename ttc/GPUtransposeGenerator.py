### __VERSION__ 40
#
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
import math
import itertools
import os
import copy
import sys

import GPUreference
import CUDAtranspose

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

###################################
#
# This file generates transpositions of the form B_perm(I) = alpha * A_I + beta * B_perm(I)
#
###################################

class GPUtransposeGenerator:
    def __init__(self, perm, loopPermutations, size, alpha, beta, maxNumImplementations,
            floatTypeA, floatTypeB, blockings, noTest, vectorLength, lda,ldb):

        self.floatTypeA = floatTypeA
        self.floatTypeB = floatTypeB
        self.cacheLineSize = 128 #in bytes

        if(floatTypeA != floatTypeB):
            print "ERROR: mixed precision has not been implemented for cuda yet."
            exit(-1)
       
        self.alpha = alpha
        self.beta = beta
        self.precision = 1e-5 # used in the function "equal" in util.cu
        self.alphaFloatType = "float"
        if( self.floatTypeA.find("double") != -1 ):
            self.alphaFloatType = "double"
	    self.precision = 1e-10

        if(self.floatTypeA == "float complex"):
             self.floatTypeA = "cuFloatComplex"
        if(self.floatTypeA == "double complex"):
             self.floatTypeA = "cuDoubleComplex"
             self.alphaFloatType = "double"
	     self.precision = 1e-10
     
       
        self.size = copy.deepcopy(size)

        self.dim = len(perm)
        self.perm = copy.deepcopy(perm)
        self.indent = "   "
        self.noTest = noTest
        self.lda = copy.deepcopy(lda)
        self.ldb = copy.deepcopy(ldb)

        self.matCopy = 0
        count = 0
        t=0 
        for p in perm:
	    if(p==count):
		t=t+1
            count = count+1
	if(count == t):
	    self.matCopy = 1
      
        self.floatSizeA = self.__getFloatTypeSize()
        
        if(len(vectorLength) == 0):
            self.vectorLength = [128,256,512]
        else:
            self.vectorLength = vectorLength
	    if(self.perm[0] != 0):	
	       available = [128,256,512]	
	       for v in vectorLength:
		   found=0
		   for a in available:
		       if(a==v):
		          found = 1
	           if(found == 0):
		      print FAIL+ "ERROR: Please choose one of the following vectorLengths: [128,256,512]" + ENDC
		      exit(-1)
	    else:
	       for v in vectorLength:
		  if(v > 1024):
		     print FAIL + "ERROR: Please choose a vector length less than 1024" + ENDC
		     exit(-1)
		       		 				 	 	 


        self.remainderA = 0
        self.remainderB = 0
        if( perm[0] != 0):
           self.remainderA = size[0] % 32
           self.remainderB = size[perm[0]] % 32
        elif(self.matCopy == 0):
           self.remainderA = size[1] % 32
           self.remainderB = size[perm[1]] % 32
	else:
           if(len(size) != 1): 
              self.remainderA = size[0] % 32
              self.remainderB = size[1] % 32


	   
        minA=32    
        self.blockings = []
	if(len(blockings) == 0):
	    self.blockings = self.getBlockings(size,perm)
        else:
	    if(self.perm[0] != 0):
	        availableBlockings = self.getBlockings(size,perm) 	
                for blocking in blockings:
                    if( blocking[0] % minA != 0 or blocking[1] % minA != 0):
                        print FAIL+ "ERROR: blockings are not a multiple of %d."%minA + ENDC
                        exit(-1)

		    found = 0	
		    for available in availableBlockings:
		        if(blocking == available): 				
                            if( self.size[0] < blocking[0] or self.size[self.perm[0]] < blocking[1] ):
				print WARNING + "Chosen blocking is higher than the tensor dimension" + ENDC
                            self.blockings.append(blocking)
                            found = 1
	
		    if(found == 0):
			print WARNING+ "WARNING: Blocking (%dx%d) is not available"%(blocking[0],blocking[1]) + ENDC	 

                if( len(self.blockings) == 0 ):
                    print WARNING+ "WARNING: none of the specified blockings are available. Reverting to default blocking (32,32)" + ENDC
                    self.blockings.append((32,32))
            else:
                if(len(size) != 1):
                   for blocking in blockings:
                      self.blockings.append(blocking)
                else:
                    print WARNING + "No Special blockings for Matrix copy" + ENDC
                    self.blockings.append([32,32])  	    		

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
        self.referenceImplementation = GPUreference.referenceTranspose( self.perm, range(len(perm))[-1::-1], self.size, self.alpha, self.beta, self.floatTypeA,self.lda,self.ldb)


        #create tmp directory or delete existing .cu files
        self.tmpDirectory = "./tmp/"

        if not os.path.exists(self.tmpDirectory):
            os.makedirs(self.tmpDirectory)
        else:
            #delete all old .cu and .h files in that folder
            for filename in os.listdir(self.tmpDirectory):
                if( filename[-3:] == ".cu" or filename[-4:] == ".cpp" or filename[-2:] == ".h" ):
                    os.remove(self.tmpDirectory+filename)

        self.minImplementationsPerFile = 64
        self.maxImplementationsPerFile = 256

       
        start = 0
        if( self.perm[0] == 0 and self.matCopy == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
            start = 1
        if( len(loopPermutations) == 0 or self.matCopy == 1):
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



    def getBlockings(self,size,perm):
	
	blockings = []
	if(self.perm[0] != 0):
	    blockings.append((32,32))
            blockings.append((64,64))
	    block = [64,64]
	    if(size[0]%64 != 0):
		block[0] = 32
	    if(size[perm[0]]%64 != 0):	
		block[1] = 32
	    #if(block[0] != 32 or block[1] != 32):
	    #    blockings.append((block[0],block[1]))
	    if(block[0] == 64 and block[1] == 64):
	        blockings.append((32,64))
	        blockings.append((64,32))
	
	elif(self.matCopy == 0):
            blockings.append((1,1))
            blockLength = [8,4,2]
	    for b in blockLength:
	        block = [b,b]	
		blockings.append((block[0],block[1]))

        else:
	    blockings.append((32,32)) 

	return blockings


    def getNumSolutions(self):
        return len(self.implementations)

    def generateVersion(self,versionStr):
        for impl in self.implementations:
            if( impl.getVersionName() == versionStr ): 
                return (self.getFastestVersion(impl), impl.getCudaTransposeHeader(0) + ";\n")
        return ""

    def generate(self):
        self.getSolutions()
        self.printMain()
        self.generateImplementations()


    def getFastestVersion(self,implementation ):
	code = "#include <cuda_runtime.h>\n"
	code = "#include <cuComplex.h>\n"
        code = "#include <complex.h>\n\n"
	code += implementation.getHostCall()
	code += implementation.getCudaImplementation()
	if(self.perm[0] !=0):
            code += implementation.getSharedTransposeKernel()
	    if(self.remainderA != 0 or self.remainderB != 0):
	        code += implementation.getRemainderTransposeKernel(32)

	return code
		


    def __getFloatTypeSize(self):
        if( self.floatTypeA == "float" ):
            return 4
        if( self.floatTypeA == "double" ):
            return 8
        if( self.floatTypeA == "float complex" ):
            return 8
        if( self.floatTypeA == "double complex" ):
            return 16



    def getSolutions(self):
        counter = 0
        #generate all implementations
        for blocking in self.blockings: 
           for vectorLength in self.vectorLength:
               if(vectorLength > blocking[0]*blocking[1] and self.perm[0] != 0):
                  continue  
               for loopPerm in self.loopPermutations:
                   counter += 1
                   sys.stdout.write("Implementations generated so far: %d\r"%counter)
                   sys.stdout.flush()
                   implementation = CUDAtranspose.cuda_transpose(self.size,self.perm,loopPerm, self.floatTypeA,blocking,vectorLength,self.beta,self.lda,self.ldb)

                   if( len(self.implementations) < self.maxNumImplementations ):
                       self.implementations.append(implementation)
                       self.implementations.sort(key=lambda tup: ttc_util.getCostLoop(tup.loopPerm, self.perm, self.size) )
                   else:
                       self.implementations.append(implementation)
                       self.implementations.sort(key=lambda tup: ttc_util.getCostLoop(tup.loopPerm, self.perm, self.size) )
                       self.implementations.pop()

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
        code +="#include <immintrin.h>\n"
        code +="#include <xmmintrin.h>\n"
        code +="#include <cuComplex.h>\n"
        code += "#include <complex.h>\n\n"
        code +="\n"

        hppCode ="#include <complex.h>\n"
        hppCode ="#include <cuComplex.h>\n"
        hppCode +="#include <stdio.h>\n"
        hppCode +="#include <float.h>\n"
        hppCode +="#include <omp.h>\n"
        hppCode +="#include <stdlib.h>\n"
        hppCode +="#include <string>\n"
        hppCode +="\n"
        hppCode += "void printMatrix2Dcomplex(const %s *A, int *size);\n"%(self.floatTypeA)
        code +="void printMatrix2Dcomplex(const %s *A, int *size)"%(self.floatTypeA)
        code +="{\n"
        code +="   for(int i=0;i < size[0]; ++i){\n"
        code +="      for(int j=0;j < size[1]; ++j){\n"
        code +="       //  printf(\"(%.2e,%.2e) \", creal(A[i + j * size[0]]), cimag(A[i + j * size[0]]));\n"
        code +="      }\n"
        code +="      printf(\"\\n\");\n"
        code +="   }\n"
        code +="   printf(\"\\n\");\n"
        code +="}\n"

        hppCode +="void restore(const %s *in, %s*out, int total_size);\n"%(self.floatTypeA,self.floatTypeA)
        code +="void restore(const %s *in, %s*out, int total_size)"%(self.floatTypeA,self.floatTypeA)
        code +="{\n"
        code +="   for(int i=0;i < total_size ; ++i){\n"
        code +="      out[i] = in[i];\n"
        code +="   }\n"
        code +="}\n"

        hppCode +="int equal(const %s *A, const %s*B, int total_size);\n"%(self.floatTypeA,self.floatTypeA)
        code +="int equal(const %s *A, const %s*B, int total_size)"%(self.floatTypeA,self.floatTypeA)
        code +="{\n"
        code +="   int error = 0;\n"
        _floatType = self.floatTypeA
        if( self.floatTypeA == "cuFloatComplex" ):
            _floatType = "float"
        if( self.floatTypeA == "cuDoubleComplex" ):
             _floatType = "double"

        if(self.floatTypeA == "cuFloatComplex" or self.floatTypeA == "cuDoubleComplex"):
            code +="   const %s *Atmp = (%s*)A;\n"%(_floatType,_floatType)
            code +="   const %s *Btmp= (%s*)B;\n"%(_floatType,_floatType)
            code +="   for(int i=0;i < 2*total_size ; ++i){\n"
        else:
            code +="   const %s *Atmp= A;\n"%self.floatTypeA
            code +="   const %s *Btmp= B;\n"%self.floatTypeA
            code +="   for(int i=0;i < total_size ; ++i){\n"

        code +="      %s Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];\n"%(_floatType)
        code +="      %s Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];\n"%(_floatType)
        code +="      %s max =  (Aabs < Babs) ? Babs : Aabs;\n"%(_floatType)
        code +="      %s diff = (Aabs - Babs);\n"%(_floatType)
        code +="      diff = (diff < 0) ? -diff : diff;\n"
        code +="      if(diff > 0){\n"
        code +="          %s relError = (diff/max);\n"%(_floatType)
        code +="          if(relError > %e){\n"%self.precision  
        code +="              //printf(\"i: %d relError: %.8e %e %e\",i,relError,Atmp[i], Btmp[i]);\n"
        code +="              //exit(0);\n"
        code +="              error += 1;\n"
        code +="          }\n"
        code +="       }\n"
        code +="    }\n"
        code +="    return (error > 0) ? 0 : 1;\n"
        code +="}\n"

        f = open(self.tmpDirectory + "util.cu",'w')
        f.write(code)
        f.close()
        f = open(self.tmpDirectory + "util.h",'w')
        f.write(hppCode)
        f.close()

    def getTotalSize(self):
        totalSize = 1
        for j in range(self.dim):
            totalSize *= self.size[j]
        return totalSize

    def printMain(self):
        code = ""
        code +="#include \"transpose.h\"\n"
        code +="#include \"util.h\"\n"
        code +="#include \"measure.h\"\n"
        code +="#include <fstream>\n"
        code +="#include <time.h>\n"
        code +="#include <immintrin.h>\n"
        code +="#include <xmmintrin.h>\n"
        code +="#include <cuComplex.h>\n"
        code += "#include <complex.h>\n\n"
        code +="#include <stdlib.h>\n"
        code +="\n"

        self.generateUtil()

        code +="\n"
        code +="int main(int argc, char** argv)\n"
        code +="{\n"

        code +="   srand(time(NULL));\n"
        code +="\n"
        code +="   double start;\n"
        code +="   int nRepeat = 4;\n"
        code +="   if(argc > 2) nRepeat = atoi(argv[2]);\n"
        code +="   int dim = %d;\n"%self.dim

        line = "   int size[] = {"
        totalSize = 1
        for i in range(self.dim):
            line += str(self.size[i])
            if i != self.dim -1:
                line += ","
            totalSize *= self.size[i]
        line += "};\n"
        code +=line

        line = "   int lda[] = {"
        totalSize = 1
        for i in range(self.dim):
            line += str(self.lda[i])
            if i != self.dim -1:
                line += ","
            totalSize *= self.lda[i]
        line += "};\n"
        code +=line

        maxSize = max(self.lda[-1] * self.size[-1], self.ldb[-1] * self.size[self.perm[-1]])
        code +="   int total_size = %d;\n"%(maxSize)
        code +="   int elements_moved = 1;\n"
        code +="\n"
        code +="   //compute total size\n"
        code +="   for(int i=0;i < dim; ++i){\n"
        code +="      elements_moved *= size[i];\n"
        code +="   }\n"
        code +="\n"
        maxNumBlocks = 0;
        for implementation in self.implementations:
            maxNumBlocks = max(implementation.getNumBlocks(), maxNumBlocks)

        code +="   %s *A, *B, *B_ref, *B_copy;\n"%(self.floatTypeA)
        code +="   double time;\n"
        if( self.floatTypeA.find("double") != -1 ):
            code +="   const double alpha = %f;\n"%(self.alpha)
            code +="   const double beta = %f;\n"%(self.beta)
        else:
            code +="   const float alpha = %f;\n"%(self.alpha)
            code +="   const float beta = %f;\n"%(self.beta)

        code +="   A = (%s *) malloc(total_size * sizeof(%s));\n"%(self.floatTypeA,self.floatTypeA)
        code +="   B_ref = (%s *) malloc(total_size * sizeof(%s));\n"%(self.floatTypeA,self.floatTypeA)
        code +="   B_copy = (%s *) malloc(total_size * sizeof(%s));\n"%(self.floatTypeA,self.floatTypeA)
        code +="   B = (%s *) malloc(total_size * sizeof(%s));\n"%(self.floatTypeA,self.floatTypeA)

        code +="   %s *A_const = A;\n"%(self.floatTypeA)
        code +="   const %s *B_copy_const = B_copy;\n"%(self.floatTypeA)
        code +="\n"
        if (self.floatTypeA == "cuFloatComplex"):
            tmpType = "float"
        if (self.floatTypeA == "cuDoubleComplex"):
            tmpType = "double"
        if(self.floatTypeA == "cuFloatComplex" or self.floatTypeA == "cuDoubleComplex"):
            code +="   %s *Atmp = (%s*) A;\n"%(tmpType,tmpType)
            code +="   %s *Btmp = (%s*) B;\n"%(tmpType,tmpType)
            code +="   for(int i=0;i < 2*total_size ; ++i){\n"
            code +="      Atmp[i] = (%s)i;\n"%(tmpType)
            code +="      Btmp[i] = (%s)i;\n"%(tmpType)
            code +="   }\n"
            code +="   for(int i=0;i < total_size ; ++i){\n"
            code +="      B_ref[i] = B[i];\n"
            code +="      B_copy[i] = B[i];\n"
            code +="   }\n"

        else:
            code +="   for(int i=0;i < total_size ; ++i){\n"
            code +="      A[i] = (%s)i;\n"%(self.floatTypeA)
            code +="      B[i] = (%s)i;\n"%(self.floatTypeA)
            code +="      B_ref[i] = B[i];\n"
            code +="      B_copy[i] = B[i];\n"
            code +="   }\n"
        code +="\n"
        code +="   /***************************************************\n"
        code +="   *make sure that all versions yield the same result\n"
        code +="   ***************************************************/\n"
        code +="   double referenceBandwidth = 0;\n"
        if( self.noTest == 0 ):
            if(self.beta != 0 ):
                code +="   %s(A_const, B_ref, size, alpha, beta);\n"%self.referenceImplementation.getReferenceHeader()
            else:
                code +="   %s(A_const, B_ref, size, alpha);\n"%self.referenceImplementation.getReferenceHeader()

        versionStr = self.referenceImplementation.getReferenceHeader()
        #time reference version
        code +="   //time reference version\n"
        code +="   if( argc == 1 || argc >= 2 && std::string(\"" + versionStr + "\").compare(argv[1]) == 0){\n"
        code +="      time = FLT_MAX;\n"
        code +="      for(int i = 0; i < nRepeat; i++){\n"
        if( self.noTest == 0 ):
            code +="         if( i < 2 )\n"
            code +="            restore(B_copy_const, B, total_size);\n"
        code +="         start = omp_get_wtime();\n"
        if(self.beta != 0 ):
            code +="         %s(A_const, B, size, alpha, beta);\n"%versionStr
        else:
            code +="         %s(A_const, B, size, alpha);\n"%versionStr
        code +="         double tmpTime = omp_get_wtime() - start;\n"
        code +="         if( tmpTime < time ) time = tmpTime;\n"
        code +="      }\n"
        if self.beta != 0:
            code +="      double bandwidth = 3. * ((double)(elements_moved * sizeof("+self.floatTypeA+")))/(1<<30)/(time);\n"
        else:
            code +="      double bandwidth = 2. * ((double)(elements_moved * sizeof("+self.floatTypeA+")))/(1<<30)/(time);\n"
        code +="      referenceBandwidth = bandwidth;\n"
        code +="      printf(\"The reference version took %f and achieved %.2f GB/s \\n\",time, bandwidth);\n"
        code +="      fflush(stdout);\n"
        code +="   }\n"

        code +="   double maxBandwidth = -1;\n"
        code +="   double maxTop1Bandwidth = -1;\n"
        code +="   double maxTop5Bandwidth = -1;\n"
        code +="   double tmpBandwidth = -1;\n"

        counter = 0

        numImplementations = len(self.implementations)
        numFiles = (numImplementations + self.minImplementationsPerFile -1)  / self.minImplementationsPerFile
        if( numFiles > 20 ):
            numFiles = (numImplementations + self.maxImplementationsPerFile -1)  / self.maxImplementationsPerFile
        numSolutionsPerFile = (numImplementations + numFiles - 1) / numFiles

        #split measurement into several files
        measureHPP = ""
        for i in range(numFiles):
            code += "   tmpBandwidth = measure%d(nRepeat, argc, argv, A_const, B, B_copy_const, B_ref, alpha, beta, total_size,elements_moved, size);\n"%(i)
            code += "   maxBandwidth = (tmpBandwidth < maxBandwidth) ? maxBandwidth : tmpBandwidth;\n"

            tmpCode = "#include \"util.h\"\n"
            tmpCode += "#include \"transpose.h\"\n\n"
 
            alphaFloatType = "float"
            if( self.floatTypeA.find("double") != -1 ):
                alphaFloatType = "double"
            header = """double measure%d(int nRepeat,
                int argc, char** argv,
                %s *A_const,
                %s * B,
                const %s * B_copy_const,
                const %s * B_ref,
                const %s alpha,
                const %s beta,
                int total_size,
                int elements_moved,
                int *size)\n"""%(i,self.floatTypeA,self.floatTypeA,self.floatTypeA,self.floatTypeA,alphaFloatType , alphaFloatType)
            tmpCode +=  header + "{\n\n"
            measureHPP += header + ";\n"
            tmpCode += "    double maxBandwidth = -1;\n"
	    tmpCode += "    %s *d_A, *d_B;\n\n"%(self.floatTypeA)
	    tmpCode += "    cudaMalloc(&d_A,total_size*sizeof(%s));\n"%(self.floatTypeA)
	    tmpCode += "    cudaMalloc(&d_B,total_size*sizeof(%s));\n"%(self.floatTypeA)
	    tmpCode += "    cudaMemcpy(d_A, A_const,total_size*sizeof(%s), cudaMemcpyHostToDevice);\n\n"%(self.floatTypeA)
            for j in range(i * numSolutionsPerFile, min(numImplementations,(i+1)*numSolutionsPerFile)):
                implementation = self.implementations[j]
                transposeName = implementation.getHeaderName(0)
                versionStr = implementation.getVersionName()

                tmpCode +="    if( argc == 1 || argc >= 2 && std::string(\"" + versionStr + "\").compare(argv[1]) == 0)\n"
                tmpCode +="    {\n"  
                tmpCode +="        double time = FLT_MAX;\n"
                tmpCode +="        for(int i = 0; i < nRepeat; i++){\n"
                if( self.noTest == 0 ):
                    tmpCode +="        if( i == 0 ){\n"
                    tmpCode +="            restore(B_copy_const, B, total_size);\n"
                    tmpCode +="            cudaMemcpy(d_B, B,total_size*sizeof(%s), cudaMemcpyHostToDevice);\n"%(self.floatTypeA) 	      
                    tmpCode +="        }\n"
                tmpCode +="        double start, tmpTime;\n"
                tmpCode +="        start = omp_get_wtime();\n"
                if(self.beta != 0):
                    tmpCode +="        %s(d_A, d_B, alpha, beta);\n"%transposeName
                else:
                    tmpCode +="        %s(d_A, d_B, alpha);\n"%transposeName
                tmpCode +="        tmpTime = omp_get_wtime() - start;\n\n"
                tmpCode +="        if( i == 0 )\n"
		tmpCode +="           cudaMemcpy(B, d_B,total_size*sizeof(%s), cudaMemcpyDeviceToHost);\n\n"%(self.floatTypeA) 
                tmpCode +="        cudaError_t err = cudaGetLastError();\n"
                tmpCode +="        if(err != cudaSuccess)\n"
                tmpCode +="            printf(\"\\nKernel ERROR : %s \\n\", cudaGetErrorString(err));\n\n"   
                tmpCode +="        if( tmpTime < time ) time = tmpTime;\n"
                if( self.noTest == 0 ):
                    tmpCode +="        if(i == 0 && !equal(B_ref, B, total_size) )\n"
                    tmpCode +="        {\n"  
                    tmpCode +="            printf(\"ERROR version "+versionStr+" doesn't give the same result (line %d)\\n\",__LINE__);\n"
                    tmpCode +="            exit(-1);\n"
                    tmpCode +="        }\n"
                tmpCode +="    }\n"
                if self.beta != 0:
                    tmpCode +="      double bandwidth = 3. * ((double)(elements_moved * sizeof("+self.floatTypeA+")))/(1<<30)/(time);\n"
                else:
                    tmpCode +="      double bandwidth = 2. * ((double)(elements_moved * sizeof("+self.floatTypeA+")))/(1<<30)/(time);\n"
                tmpCode +="      if( bandwidth > maxBandwidth ) maxBandwidth = bandwidth;\n"

                tmpCode +="      printf(\"variant "+versionStr+" took %f and achieved %.2f GB/s (theoretical cost: 10) (tlb misses: 0) (l2 misses: 0.000000) (invalidates: 0.000000)\\n\",time, bandwidth);\n"
                tmpCode +="      fflush(stdout);\n"
                tmpCode +="   }\n\n\n"
                counter += 1
            tmpCode +="   return maxBandwidth;\n"
            tmpCode +="}\n"
            f = open(self.tmpDirectory + "measure%d.cu"%i,'w')
            f.write(tmpCode)
            f.close()
            f = open(self.tmpDirectory + "measure.h",'w')
            f.write(measureHPP)
            f.close()

        code +="   /***************************************************/\n"

        code +="   printf(\"Maximal bandwidth: %f\\n\", maxBandwidth);\n"
        code +="   printf(\"Speedup over reference: %f\\n\", maxBandwidth / referenceBandwidth );\n"
        code +="   printf(\"Top-1 speedup: %.2f\\n\", maxTop1Bandwidth/maxBandwidth);\n"
        code +="   printf(\"Top-5 speedup: %.2f\\n\", maxTop5Bandwidth/maxBandwidth);\n"
        code +="   printf(\"SUCCESS!\\n\");\n"
        code +="   free(A); free(B);\n"
        code +="   free(B_ref);\n"
        code +="   return 0;\n"
        code +="}\n"
        f = open(self.tmpDirectory + "main.cu",'w')
        f.write(code)
        f.close()



    def generateImplementations(self):
        #generate CPP and HPP files
        hppCode ="#include <cuComplex.h>\n"
        hppCode += "void " + self.referenceImplementation.getReferenceHeader(1) + ";\n"

	for v in self.vectorLength:
	    cudaTranspose = CUDAtranspose.cuda_transpose(self.size,self.perm,self.loopPermutations[0], self.floatTypeA,self.blockings[0],v,self.beta,self.lda,self.ldb)
	    cudaCode = "#include <cuda_runtime.h>\n"
	    cudaCode += "#include <cuComplex.h>\n"
            ##code = "#include <complex.h>\n\n"
            if(self.perm[0] != 0):
                cudaCode += cudaTranspose.getSharedTransposeKernel()
                if(self.remainderA != 0 or self.remainderB != 0):
	            cudaCode += cudaTranspose.getRemainderTransposeKernel(32)
	    for b in self.blockings:
	        for l in self.loopPermutations:
		     cudaTranspose = CUDAtranspose.cuda_transpose(self.size,self.perm,l, self.floatTypeA,b,v,self.beta, self.lda,self.ldb)       
	             cudaCode += "\n//**********************************************************************\n\n" + cudaTranspose.getCudaImplementation()
                     cudaCode += cudaTranspose.getHostCall()
		     hppCode += cudaTranspose.getCudaTransposeHeader(0) + ";\n"
            f = open(self.tmpDirectory+"cuTranspose_Vec%d.cu"%(v),'w')
            f.write(cudaCode)
            f.close()

	f = open(self.tmpDirectory+"transpose.h",'w')
        f.write(hppCode)
        f.close()


        refCode = ""
        refCode += "#include <stdio.h>\n"
        refCode += "#include <omp.h>\n"
        refCode += "#include <cuComplex.h>\n"      
        refCode += "\n" 
        refCode += self.referenceImplementation.generateReferenceImplementation()

        f = open(self.tmpDirectory+"reference.cu",'w')
        f.write(refCode)
        f.close()


