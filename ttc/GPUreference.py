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


class referenceTranspose:
    def __init__(self,perm,loopPerm,size,alpha,beta,floatTypeA,floatTypeB,lda,ldb):
        
	self.perm = copy.deepcopy(perm)
	self.loopPerm = copy.deepcopy(loopPerm)
	self.size = copy.deepcopy(size)
	self.alpha = alpha
	self.beta = beta
	self.floatTypeA = floatTypeA
        self.floatTypeB = floatTypeB

        self.alphaFloatType = "float"
        if( self.floatTypeA.find("double") != -1 ):
            self.alphaFloatType = "double"
        if(self.floatTypeA == "cuDoubleComplex"):
            self.alphaFloatType = "double"

        self.betaFloatType = "float"
        if( self.floatTypeB.find("double") != -1 ):
            self.betaFloatType = "double"
        if(self.floatTypeB == "cuDoubleComplex"):
            self.betaFloatType = "double"

	self.indent = "   "
        self.dim = len(perm)
        #compute leading dimensions
        self.lda = copy.deepcopy(lda)
        self.ldb = copy.deepcopy(ldb)

	self.cost = ttc_util.getCostLoop(self.loopPerm, self.perm, self.size)
	self.code=""

    def getOffsetA(self):
        offset = ""
        for i in range(self.dim):
            offset += "i" + str(i)
            if(self.lda[i] != 1):
                offset += "*lda" + str(i)
            if( i != self.dim-1):
                offset += " + "
        return offset

    def getOffsetB(self):
        offset = ""
        for i in range(self.dim):
            #find idx idxPerm
            invIdx = -1
            for j in range(self.dim):
                if self.perm[j] == i:
                    invIdx = j
            offset += "i" + str(i)
            if(self.ldb[invIdx] != 1):
                offset += "*ldb" + str(invIdx)
            if( i != self.dim-1):
                offset += " + "
        return offset

    def getUpdateString(self,indent):
        outStr = "B[" + self.getOffsetB() + "]"

        inStr = "A[" + self.getOffsetA() + "]"

        ret = ""

        if(self.floatTypeA == "float" or self.floatTypeA == "double"):
           if(self.beta != 0):
              ret +=  "%s%s = alpha*%s + beta*%s;\n"%(indent + self.indent, outStr, inStr,outStr)
           else:
              ret +=  "%s%s = alpha*%s;\n"%(indent + self.indent, outStr, inStr)
        
        if(self.floatTypeA == "cuFloatComplex"):
          CuMulA = "cuCmulf"
        if(self.floatTypeA == "cuDoubleComplex"):
          CuMulA = "cuCmul"  
        if(self.floatTypeB == "cuFloatComplex"):
          CuMulB = "cuCmulf"
        if(self.floatTypeB == "cuDoubleComplex"):
          CuMulB = "cuCmul" 
        mixed = 1
        if(self.floatTypeA == self.floatTypeB):
            mixed = 0
        
        if(self.floatTypeB == "cuFloatComplex" and mixed == 0):
           if(self.beta != 0):
              ret +=  "%s%s = cuCaddf(%s(cuAlpha,%s) , %s(cuBeta,%s));\n"%(indent + self.indent, outStr,CuMulA, inStr,CuMulB,outStr)
           else:
              ret +=  "%s%s = %s(cuAlpha,%s);\n"%(indent + self.indent, outStr, CuMulA, inStr)
        if(self.floatTypeB == "cuDoubleComplex" and mixed == 0):
           if(self.beta != 0):
              ret +=  "%s%s = cuCadd(%s(cuAlpha,%s) , %s(cuBeta,%s));\n"%(indent + self.indent, outStr,CuMulA, inStr,CuMulB,outStr)
           else:
              ret +=  "%s%s = %s(cuAlpha,%s);\n"%(indent + self.indent, outStr,CuMulA, inStr)

        if(self.floatTypeB == "cuFloatComplex" and mixed == 1):
           if(self.beta != 0):
              ret +=  "%s%s = cuCaddf(cuComplexDoubleToFloat(%s(cuAlpha,%s)) , %s(cuBeta,%s));\n"%(indent + self.indent, outStr,CuMulA, inStr,CuMulB,outStr)
           else:
              ret +=  "%s%s = cuComplexDoubleToFloat(%s(cuAlpha,%s));\n"%(indent + self.indent, outStr, CuMulA, inStr)
        if(self.floatTypeB == "cuDoubleComplex" and mixed == 1):
           if(self.beta != 0):
              ret +=  "%s%s = cuCadd(cuComplexFloatToDouble(%s(cuAlpha,%s)) , %s(cuBeta,%s));\n"%(indent + self.indent, outStr,CuMulA, inStr,CuMulB,outStr)
           else:
              ret +=  "%s%s = cuComplexFloatToDouble(%s(cuAlpha,%s));\n"%(indent + self.indent, outStr,CuMulA, inStr)

        return ret


    def printReferenceLoop(self, loopPerm, indent):
        loopIdx = loopPerm[0]
        increment = 1
        self.code +=  "%sfor(int i%d = 0; i%d < size%d; i%d += %d)\n"%(indent,loopIdx,loopIdx,loopIdx,loopIdx,increment)

        if len(loopPerm) > 1:
            self.printReferenceLoop(loopPerm[1:], indent + self.indent)
        else:#we reached the innermost loop, no recursion

            #get input and output offsets correct
            self.code += self.getUpdateString(indent)	

    def getPermVersion(self):
	code =""
	for i in self.perm:
		code += str(i)
	return code

    def getReferenceHeader(self,signature=0):
        if(self.floatTypeA == "float"):
            tmpChar = "s"
        if(self.floatTypeA == "double"):
            tmpChar = "d"
        elif(self.floatTypeA == "cuFloatComplex"):
            tmpChar = "c"
        elif(self.floatTypeA == "cuDoubleComplex"):
            tmpChar = "z"

        if(self.floatTypeA != self.floatTypeB):
           if(self.floatTypeB == "float"):
               tmpChar += "s"
           if(self.floatTypeB == "double"):
               tmpChar += "d"
           elif(self.floatTypeB == "cuFloatComplex"):
               tmpChar += "c"
           elif(self.floatTypeB == "cuDoubleComplex"):
               tmpChar += "z"

        transposeName = "%sCuTranspose_"%tmpChar
        
        for i in self.perm:
            transposeName += str(i)

	transposeName +="_"
        for idx in range(len(self.size)):
             transposeName += "%d"%(self.size[idx])
             if(idx != len(self.size)-1):
		 transposeName +="x"

	transposeName +="_reference"
	if(self.beta == 0):
	    transposeName +="_bz"

	code = ""
        code += "%s"%(transposeName)
	if(signature):
	    if(self.beta == 0): 
	        code += "(const %s *A, %s *B, int* sizeA, const %s alpha, const int *lda, const int *ldb)"%(self.floatTypeA, self.floatTypeB,self.alphaFloatType)
	    else:
		code += "(const %s *A, %s *B, int* sizeA, const %s alpha, const %s beta, const int *lda, const int *ldb)"%(self.floatTypeA, self.floatTypeB,self.alphaFloatType, self.betaFloatType) 	
	return code


    def generateReferenceImplementation(self):

        self.code += "//Loop-order: " + str(self.loopPerm) + " Theoretical cost: " + str(self.cost) + "\n"
        
        self.code += "void %s\n{\n\n"%(self.getReferenceHeader(1))

        self.code += self.declareVariables()
        self.code += "#pragma omp parallel for collapse(%d)\n"%(len(self.loopPerm)-1)
        self.printReferenceLoop(self.loopPerm, self.indent)
        self.code += "}\n"

	return self.code


    def declareVariables(self):
	code=""
        for i in range(self.dim):
            code +=  "%sconst int size%d = %d;\n"%(self.indent,i,self.size[i])
        for i in range(len(self.lda)):
            if( i == 0):
               code +=  "%sconst int lda0 = 1;\n"%(self.indent)
            else:
               code +=  "%sconst int lda%d = lda%d*lda[%d];\n"%(self.indent,i,i-1,i-1)
        for i in range(len(self.ldb)):
            if( i == 0):
               code +=  "%sconst int ldb0 = 1;\n"%(self.indent)  
            else:      
               code +=  "%sconst int ldb%d = ldb%d*ldb[%d];\n"%(self.indent,i,i-1,i-1)

        if(self.floatTypeA == "cuFloatComplex"):
            code +=   "%scuFloatComplex cuAlpha = make_cuFloatComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuFloatComplex"):
            if(self.beta != 0):
                code +=   "%scuFloatComplex cuBeta = make_cuFloatComplex(beta,0.0);\n"%self.indent

        if(self.floatTypeA == "cuDoubleComplex"):
            code +=   "%scuDoubleComplex cuAlpha = make_cuDoubleComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuDoubleComplex"):
            if(self.beta != 0):
                code +=   "%scuDoubleComplex cuBeta = make_cuDoubleComplex(beta,0.0);\n"%self.indent

	return code



