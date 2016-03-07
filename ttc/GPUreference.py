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
    def __init__(self,perm,loopPerm,size,alpha,beta,floatType,lda,ldb):
        
	self.perm = copy.deepcopy(perm)
	self.loopPerm = copy.deepcopy(loopPerm)
	self.size = copy.deepcopy(size)
	self.alpha = alpha
	self.beta = beta
	self.floatType = floatType
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

        if(self.floatType == "float" or self.floatType == "double"):
           if(self.beta != 0):
              ret +=  "%s%s = alpha*%s + beta*%s;\n"%(indent + self.indent, outStr, inStr,outStr)
           else:
              ret +=  "%s%s = alpha*%s;\n"%(indent + self.indent, outStr, inStr)
        if(self.floatType == "cuFloatComplex"):
           if(self.beta != 0):
              ret +=  "%s%s = cuCaddf(cuCmulf(cuAlpha,%s) , cuCmulf(cuBeta,%s));\n"%(indent + self.indent, outStr, inStr,outStr)
           else:
              ret +=  "%s%s = cuCmulf(cuAlpha,%s);\n"%(indent + self.indent, outStr, inStr)
        if(self.floatType == "cuDoubleComplex"):
           if(self.beta != 0):
              ret +=  "%s%s = cuCadd(cuCmul(cuAlpha,%s) , cuCmul(cuBeta,%s));\n"%(indent + self.indent, outStr, inStr,outStr)
           else:
              ret +=  "%s%s = cuCmul(cuAlpha,%s);\n"%(indent + self.indent, outStr, inStr)

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
        alphaFloatType = "float"
        if( self.floatType.find("double") != -1 ):
            alphaFloatType = "double"

        if(self.floatType == "float"):
            transposeName = "sCuTranspose_"
        elif(self.floatType == "double"):
            transposeName = "dCuTranspose_"
        elif(self.floatType == "cuFloatComplex"):
            transposeName = "cCuTranspose_"
        elif(self.floatType == "cuDoubleComplex"):
            transposeName = "zCuTranspose_"
        
        for i in self.perm:
            transposeName += str(i)

	transposeName +="_"
        for idx in range(len(self.size)):
             transposeName += "%d"%(self.size[idx])
             if(idx != len(self.size)-1):
		 transposeName +="x"

        transposeName +="_"
        for idx in range(len(self.lda)):
             transposeName += "%d"%(self.lda[idx])
             if(idx != len(self.lda)-1):
		 transposeName +="x"

	transposeName +="_"
        for idx in range(len(self.ldb)):
             transposeName += "%d"%(self.ldb[idx])
             if(idx != len(self.ldb)-1):
		 transposeName +="x"

	transposeName +="_reference"
	if(self.beta == 0):
	    transposeName +="_bz"

	code = ""
        code += "%s"%(transposeName)
	if(signature):
	    if(self.beta == 0): 
	        code += "(const %s *A, %s *B, int* sizeA, const %s alpha)"%(self.floatType, self.floatType,alphaFloatType)
	    else:
		code += "(const %s *A, %s *B, int* sizeA, const %s alpha, const %s beta)"%(self.floatType, self.floatType,alphaFloatType, alphaFloatType) 	
	return code


    def generateReferenceImplementation(self):

        self.code += "//Loop-order: " + str(self.loopPerm) + " Theoretical cost: " + str(self.cost) + "\n"

        alphaFloatType = "float"
        if( self.floatType.find("double") != -1 ):
            alphaFloatType = "double"
        
        self.code += "void %s\n{\n\n"%(self.getReferenceHeader(1))

        self.code += self.declareVariables()
        self.code += "#pragma omp parallel for collapse(%d)\n"%(len(self.loopPerm)-1)
        self.printReferenceLoop(self.loopPerm, self.indent)
        self.code += "}\n"

	return self.code


    def declareVariables(self):
	code=""
        for i in range(self.dim):
            code +=  "%sconst int size%d = sizeA[%d];\n"%(self.indent,i,i)
        for i in range(len(self.lda)):
            code +=  "%sconst int lda%d = %d;\n"%(self.indent,i,self.lda[i])
        for i in range(len(self.ldb)):
            code +=  "%sconst int ldb%d = %d;\n"%(self.indent,i,self.ldb[i])

        if(self.floatType == "cuFloatComplex"):
            code +=   "%scuFloatComplex cuAlpha = make_cuFloatComplex(alpha,0.0);\n"%self.indent
            if(self.beta != 0):
                code +=   "%scuFloatComplex cuBeta = make_cuFloatComplex(beta,0.0);\n"%self.indent

        if(self.floatType == "cuDoubleComplex"):
            code +=   "%scuDoubleComplex cuAlpha = make_cuDoubleComplex(alpha,0.0);\n"%self.indent
            if(self.beta != 0):
                code +=   "%scuDoubleComplex cuBeta = make_cuDoubleComplex(beta,0.0);\n"%self.indent

	return code



