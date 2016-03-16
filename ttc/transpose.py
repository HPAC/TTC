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

import itertools
import copy
import time 
import ttc_util


###################################
#
# This file generates transpositions of the form B_perm(I) = alpha * A_I + beta * B_perm(I)
#
###################################


class implementation:
    def __init__(self, blocking, loopPerm, perm, size, alpha, beta, floatTypeA, floatTypeB,
            optimization, scalar, prefetchDistance, microBlocking, reference,
            architecture, parallelize):

        self.parallelize = parallelize
        self.debug = 0 
        self.floatTypeA = floatTypeA
        self.floatTypeB = floatTypeB
        self.architecture = architecture
        self.alpha = alpha
        self.beta = beta 
        self.optimization = optimization #TODO remane optimization to streamingStores
        self.prefetchDistance = prefetchDistance
        self.microBlocking = microBlocking
        self.numMicroBlocksPerBlock = blocking[0]/microBlocking[0] * blocking[1]/microBlocking[1]
        self.reference = reference

        self.size = copy.deepcopy(size)

        self.scalar = scalar

        if self.reference == 0:
            self.blockA= blocking[0] #blocking in stride-1 indices of A 
            self.blockB= blocking[1] #blocking in stride-1 indices of B
        else:
            self.blockA= 1 #blocking in stride-1 indices of A 
            self.blockB= 1 #blocking in stride-1 indices of B

        #deal with remainder
        self.remainderA = 0
        self.remainderB = 0
        if( perm[0] != 0):
            self.remainderA = size[0] % self.blockA
            self.remainderB = size[perm[0]] % self.blockB
        else:
            if( len(self.size) == 1):
                self.remainderA = 0
                self.remainderB = 0
            else:
                self.remainderA = size[1] % self.blockA
                self.remainderB = size[perm[1]] % self.blockB

        self.dim = len(perm)

        self.perm = copy.deepcopy(perm)
        self.loopPerm = copy.deepcopy(loopPerm)

        self.indent = "   "
        self.cost = 0.0

        self.code = ""

        self.ldout = -1
        for i in range(len(self.perm)):
            if( self.perm[0] != 0 ):
                if self.perm[i] == 0:
                    self.ldout = i
                    break;
            else:
                if self.perm[i] == 1:
                    self.ldout = i
                    break;

        self.transposeMacroKernelname = "transpose%dx%d"%(self.blockA,self.blockB)
        if( self.optimization != "" ):
            self.transposeMacroKernelname += "_"+ self.optimization

        if( self.prefetchDistance > 0 ):
            self.transposeMacroKernelname += "_prefetch_"+ str(self.prefetchDistance)

    def getPrefetchDistance(self):
        return self.prefetchDistance 

    def getLoopPerm(self):
        return self.loopPerm

    def getOffsetA(self,start = 0):
        offset = ""
        for i in range(start,self.dim):
            offset += "i" + str(i)
            if(i != 0):
                offset += "*lda" + str(i)
            if( i != self.dim-1):
                offset += " + "
        return offset 

    def getOffsetB(self, start = 0):
        offset = ""
        for i in range(start,self.dim):
            #find idx idxPerm
            invIdx = -1
            for j in range(self.dim):
                if self.perm[j] == i:
                    invIdx = j
            offset += "i" + str(i)
            if(invIdx != 0):
                offset += "*ldb" + str(invIdx)
            if( i != self.dim-1):
                offset += " + "
        return offset

    def getUpdateString(self,indent):
        outStr = "B[" + self.getOffsetB() + "]"

        inStr = "A[" + self.getOffsetA() + "]"

        ret = ""

        if(self.beta != 0):
            ret +=  "%s%s = alpha*%s + beta*%s;\n"%(indent + self.indent, outStr, inStr,outStr)
        else:
            ret +=  "%s%s = alpha*%s;\n"%(indent + self.indent, outStr, inStr)

        return ret


    def printScalarLoop(self, loopPerm, indent):
        loopIdx = loopPerm[0]
        increment = 1
        if len(loopPerm) == 1: 
            self.code += "#pragma simd\n"
        self.code +=  "%sfor(int i%d = 0; i%d < size%d; i%d += %d)\n"%(indent,loopIdx,loopIdx,loopIdx,loopIdx,increment)

        if len(loopPerm) > 1: 
            self.printScalarLoop(loopPerm[1:], indent + self.indent)
        else:#we reached the innermost loop, no recursion

            #get input and output offsets correct
            self.code += self.getUpdateString(indent)


    def printRemainderLoop(self, loopPerm, indent, remainderIdx):
        loopIdx = loopPerm[0]
        increment = 1
        if(loopIdx == remainderIdx):
            self.code +=  "%sfor(int i%d = size%d - remainder%d; i%d < size%d; i%d += %d)\n"%(indent,loopIdx, loopIdx, loopIdx,loopIdx,loopIdx,loopIdx,increment)
        else:
            firstIdx = 0
            if( self.perm[0] == 0 ):
                firstIdx = 1
            if( remainderIdx == firstIdx and self.remainderB != 0 and loopIdx == self.perm[firstIdx]):
                self.code +=  "%sfor(int i%d = 0; i%d < size%d - remainder%d; i%d += %d)\n"%(indent,loopIdx,loopIdx,loopIdx,loopIdx,loopIdx,increment)
            else:
                self.code +=  "%sfor(int i%d = 0; i%d < size%d; i%d += %d)\n"%(indent,loopIdx,loopIdx,loopIdx,loopIdx,increment)

        if len(loopPerm) > 1: 
            self.printRemainderLoop(loopPerm[1:], indent + self.indent, remainderIdx)
        else:#we reached the innermost loop, no recursion
            if( self.perm[0] == 0 ):
                indent += self.indent
                self.code +=  "%sfor(int i0 = 0; i0 < size0; i0++)\n"%(indent)

            #get input and output offsets correct
            self.code += self.getUpdateString(indent)

    def getBlocking(self):
        return (self.blockA, self.blockB)

    def getBroadcastVariables(self):
        code = " "
        if(self.perm[0]!=0 and self.scalar == 0): 
            code += " ,reg_alpha"  
            if(self.beta):
                code += " ,reg_beta"
        else:
            code += " ,alpha"
            if(self.beta):
                code += " ,beta"

        return code 

    def __printLoopBody(self, loopPerm, indent):
        loopIdx = loopPerm[0]
        increment = 1
        if( self.perm[0] != 0 ):
            if loopIdx == 0: 
                increment = self.blockA
            elif loopIdx  == self.perm[0]:
                increment = self.blockB
        else:
            #we block along the outer two dimensions if the first index doesn't change
            if loopIdx == 1: 
                increment = self.blockA
            elif loopIdx  == self.perm[1]:
                increment = self.blockB

        if( increment > 1):
            self.code +=  "%sfor(int i%d = 0; i%d < size%d - %d; i%d+= %d)\n"%(indent,loopIdx,loopIdx,loopIdx,increment-1,loopIdx,increment)
        else:
            self.code +=  "%sfor(int i%d = 0; i%d < size%d; i%d+= %d)\n"%(indent,loopIdx,loopIdx,loopIdx,loopIdx,increment)

        if len(loopPerm) > 1: #we have not reached the inner most loop yet => recursion
            self.__printLoopBody(loopPerm[1:], indent + "   ")
        else: #we reached the innermost loop, no recursion

            if( self.prefetchDistance > 0):
                indexStr = ""
                indexPrintStr = "("
                for i in range(self.dim):
                    indexStr += "i%d, "%i
                    indexPrintStr += "%d, "
                indexPrintStr += ")"
                self.code += "%s{\n"%indent
                self.code += "%sint offsetA = %s;\n"%(indent + self.indent, self.getOffsetA())
                self.code += "%sint offsetB = %s;\n"%(indent + self.indent, self.getOffsetB())
                
                prefetchDistance = (self.prefetchDistance + self.numMicroBlocksPerBlock - 1) / self.numMicroBlocksPerBlock
                self.code += "%sif( counter >= %d ){\n"%(indent + self.indent, prefetchDistance )
                self.code += "%sconst Offset &task = tasks.back();\n"%(indent + self.indent + self.indent)

                self.code += "%sint offsetAnext0 = task.offsetA;\n"%(indent + self.indent + self.indent)
                self.code += "%sint offsetBnext0 = task.offsetB;\n"%(indent + self.indent + self.indent)

                self.code += "%sconst Offset &currentTask = tasks.front();\n"%(indent + self.indent + self.indent)
                self.code += "%s%s(&A[currentTask.offsetA], lda%d, &B[currentTask.offsetB], ldb%d, &A[offsetAnext0], &B[offsetBnext0], &A[offsetA], &B[offsetB]%s);\n"%(indent + self.indent + self.indent, self.transposeMacroKernelname, self.perm[0], self.ldout, self.getBroadcastVariables())
                self.code += "%stasks.pop();\n"%(indent + self.indent + self.indent)
                self.code += "%s}\n"%(indent + self.indent)

                self.code += "%scounter++;\n"%(indent + self.indent)
                self.code += "%sOffset offset; offset.offsetA = offsetA; offset.offsetB = offsetB;\n"%(indent + self.indent)
                self.code += "%stasks.push( offset );\n"%(indent + self.indent)

                #if self.debug: 
                #    self.code += "%sif( offsetA != offsetAnext || offsetB != offsetBnext)\n"%(indent + self.indent)
                #    self.code += "%s   printf(\"%%d: %s %s %%d %%d %%d %%d\\n\",omp_get_thread_num(), %soffsetA, offsetAnext1, offsetB, offsetBnext1);\n"%(indent + self.indent,self.getVersionName(), indexPrintStr, indexStr)
                #    self.code += "%soffsetAnext = offsetAnext1;\n"%(indent)
                #    self.code += "%soffsetBnext = offsetBnext1;\n"%(indent)

            else:
                if( self.perm[0] != 0):
                    self.code +=  "%s%s(&A[%s], lda%d, &B[%s], ldb%d%s);\n"%(indent + self.indent, self.transposeMacroKernelname,self.getOffsetA(), self.perm[0], self.getOffsetB(),self.ldout, self.getBroadcastVariables())
                else:
                    self.code +=  "%s%s(&A[%s], lda1, lda%d, &B[%s], ldb1, ldb%d%s);\n"%(indent + self.indent, self.transposeMacroKernelname,self.getOffsetA(1), self.perm[1], self.getOffsetB(1),self.ldout, self.getBroadcastVariables())

            if( self.prefetchDistance > 0 ):
                self.code += "%s}\n"%indent


    def getVersionName(self):
        versionName = ""
        if(self.reference != 0):
            versionName += "reference"
        else:
            versionName += "v"
            found0 = 0
            for i in self.loopPerm:
                if(i == 0):
                    found0 = 1
                versionName += str(i)
            if(self.perm[0] == 0 and not found0):
                versionName += str(0) #0 is always the innermost loop in this case

            versionName += "_%dx%d"%(self.blockA, self.blockB)
            if( self.optimization != "" ):
                versionName += "_"+self.optimization
            if( self.prefetchDistance > 0 ):
                versionName += "_prefetch_" + str(self.prefetchDistance)


        return versionName

    def getTransposeName(self, clean = 0):
        if(self.floatTypeA == "float"):
            if(self.floatTypeB == "float"):
                transposeName = "s"
            else:
                transposeName = "sd"
        if(self.floatTypeA == "double"):
            if(self.floatTypeB == "double"):
                transposeName = "d"
            else:
                transposeName = "ds"
        if(self.floatTypeA == "float complex"):
            if(self.floatTypeB == "float complex"):
                transposeName = "c"
            else:
                transposeName = "cz"
        if(self.floatTypeA == "double complex"):
            if(self.floatTypeB == "double complex"):
                transposeName = "z"
            else:
                transposeName = "zs"
        transposeName += "Transpose_"

     
        for i in self.perm:
            transposeName += str(i)
  
	transposeName +="_"
        for idx in range(len(self.size)):
             transposeName += "%d"%(self.size[idx])
             if(idx != len(self.size)-1):
		 transposeName +="x"

#        transposeName +="_"
#        for idx in range(len(self.lda)):
#             transposeName += "%d"%(self.lda[idx])
#             if(idx != len(self.lda)-1):
#		 transposeName +="x"
#
#	transposeName +="_"
#        for idx in range(len(self.ldb)):
#             transposeName += "%d"%(self.ldb[idx])
#             if(idx != len(self.ldb)-1):
#		 transposeName +="x"

        if(clean == 0):
            transposeName += "_"
            transposeName += self.getVersionName()

        if(self.parallelize == 1):
            transposeName += "_par"

        if(self.beta == 0):
            transposeName += "_bz"

        return transposeName


    def getBroadcastKernel(self, name, value, floatType):
        self.code += self.indent +"//broadcast %s\n"%name

        if(self.architecture == "power"):
            self.code += self.indent + "vector4double %s = vec_splats(%s);\n"%(name, value)
        else:
            if( self.architecture == "knc" or self.architecture == "avx512" ):
                registerSizeBits = 512
            else:
                registerSizeBits = 256

            if( value == "beta" and self.floatTypeA.find("double") != -1 and self.floatTypeB.find("float") != -1):
                _floatType = "__m128"
                functionName = "_mm_set1_ps"
            elif( floatType == "float" or floatType == "float complex" ):
                functionName = "_mm%d_set1_ps"%registerSizeBits
                _floatType = "__m%d"%registerSizeBits
            elif( floatType == "double" or  floatType == "double complex" ):
                functionName = "_mm%d_set1_pd"%registerSizeBits
                _floatType = "__m%dd"%registerSizeBits

            self.code += self.indent + "%s %s = %s(%s);\n"%(_floatType, name,functionName,value)
        return self.code + "\n"

    def getHeader(self, headerFlag = 1, clean = 0):

        transposeName = self.getTransposeName(clean)
        if headerFlag == 0:
            trailingChar = "\n{\n"
        else:
            trailingChar = ";\n"
        
        alphaFloatType = "float"
        if( self.floatTypeA.find("double") != -1 ):
            alphaFloatType = "double"
        betaFloatType = "float"
        if( self.floatTypeB.find("double") != -1 ):
            betaFloatType = "double"
        if(self.beta != 0): 
            return "void %s( const %s* __restrict__ A, %s* __restrict__ B, const %s alpha, const %s beta, const int *size, const int *lda, const int *ldb)%s"% (transposeName, self.floatTypeA, self.floatTypeB, alphaFloatType,betaFloatType, trailingChar)
        else: 
            return "void %s( const %s* __restrict__ A, %s* __restrict__ B, const %s alpha, const int *size, const int *lda, const int *ldb)%s"% (transposeName, self.floatTypeA, self.floatTypeB, alphaFloatType, trailingChar)


    def printHeader(self, headerFlag = 1, clean = 0):
        self.code +=  self.getHeader(headerFlag, clean)

    def declareVariables(self):
        for i in range(self.dim):
            self.code +=  "%sconst int size%d = size[%d];\n"%(self.indent,i,i)
        for i in range(self.dim):
            self.code +=  "%sconst int lda%d = lda[%d];\n"%(self.indent,i,i)
        for i in range(self.dim):
            self.code +=  "%sconst int ldb%d = ldb[%d];\n"%(self.indent,i,i)
        if( self.perm[0] != 0 ):
            if( self.remainderA != 0):
                self.code +=   "%sconst int remainder0 = %d;\n"%(self.indent,self.remainderA)
            if( self.remainderB != 0):
                self.code +=   "%sconst int remainder%d = %d;\n"%(self.indent,self.perm[0], self.remainderB)
        else:
            if( self.remainderA != 0):
                self.code +=   "%sconst int remainder1 = %d;\n"%(self.indent,self.remainderA)
            if( self.remainderB != 0):
                self.code +=   "%sconst int remainder%d = %d;\n"%(self.indent,self.perm[1], self.remainderB)
        if( self.prefetchDistance > 0 and self.debug ):
            self.code +=   "%sint offsetAnext = 0, offsetBnext = 0;\n"%(self.indent)


    def getCostLoop(self):

        if( self.cost != 0.0 ):
            return self.cost

        self.cost = ttc_util.getCostLoop(self.loopPerm, self.perm, self.size)
        return self.cost


    def getImplementation(self, parallel = 1, clean = 0):

        self.code = ""

        if( clean ):
            self.printHeader(0,1)
        else:
            self.printHeader(0)
        self.declareVariables()

        if(self.perm[0] != 0 and self.scalar ==0):  
            self.getBroadcastKernel("reg_alpha","alpha", self.floatTypeA)
            if(self.beta != 0):
                self.getBroadcastKernel("reg_beta","beta", self.floatTypeB)		


        if( self.reference == 0):
            indent = self.indent
            if( parallel ):
               self.code += "#pragma omp parallel\n"
               self.code += self.indent +"{\n"
               indent += self.indent
            if( self.prefetchDistance > 0 ):
               self.code += indent + "int counter = 0;\n"
               self.code += indent + "std::queue<Offset> tasks;\n"
            
            if( parallel ):
               self.code += "#pragma omp for collapse(%d) schedule(static)\n"%(len(self.loopPerm))
            self.__printLoopBody(self.loopPerm, indent)

            if( self.prefetchDistance > 0 ):
                self.code += indent + "while(tasks.size() > 0){\n"
                self.code += indent + "    const Offset &task = tasks.front();\n"
                endPos = self.transposeMacroKernelname.find("prefetch")
                if( endPos != -1):
                    endPos -= 1 #remove last '_'
                cleanMacroTransposeName = self.transposeMacroKernelname[:endPos]#remove prefetch
                self.code += indent + "    %s(&A[task.offsetA], lda%d, &B[task.offsetB], ldb%d %s);\n"%(cleanMacroTransposeName, self.perm[0], self.ldout, self.getBroadcastVariables())
                self.code += indent + "    tasks.pop();\n"
                self.code += indent + "}\n"
            #print remainder loops
            indent = self.indent
            if( parallel ):
                indent += self.indent
            if( self.perm[0] != 0 ):
                if( self.remainderA != 0 and self.size[self.perm[0]] - self.remainderB > 0):
                    if( parallel ):
                        self.code += "#pragma omp for collapse(%d) schedule(static)\n"%(self.dim)
                    self.code += indent + "//Remainder loop" + "\n"
                    self.printRemainderLoop(self.loopPerm, indent, 0)

                if( self.remainderB != 0 ):
                    if( parallel ):
                        self.code += "#pragma omp for collapse(%d) schedule(static)\n"%(self.dim)
                    self.code += indent + "//Remainder loop" + "\n"
                    self.printRemainderLoop(self.loopPerm, indent, self.perm[0])
            else:
                if( self.remainderA != 0 and self.size[self.perm[1]] - self.remainderB > 0):
                    if( parallel ):
                        self.code += "#pragma omp for collapse(%d) schedule(static)\n"%(self.dim-1)
                    self.code += indent + "//Remainder loop" + "\n"
                    self.printRemainderLoop(self.loopPerm, indent, 1)

                if( self.remainderB != 0 ):
                    if( parallel ):
                        self.code += "#pragma omp for collapse(%d) schedule(static)\n"%(self.dim-1)
                    self.code += indent + "//Remainder loop" + "\n"
                    self.printRemainderLoop(self.loopPerm, indent, self.perm[1])

            if( parallel ):
                self.code += self.indent +"}\n"
        else:
            if( parallel ):
               self.code += "#pragma omp parallel for collapse(%d)\n"%(max(1,len(self.loopPerm)-1))
            self.printScalarLoop(self.loopPerm, self.indent)

        self.code += "}\n"
        return self.code

