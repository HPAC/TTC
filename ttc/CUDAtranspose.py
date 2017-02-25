import copy
import math
import ttc_util

class cuda_transpose:
    def __init__(self,size,perm,loopPerm, floatTypeA, floatTypeB, blocking, vectorLength,isBeta,lda,ldb):


       self.size = copy.deepcopy(size)

       self.perm = copy.deepcopy(perm)
       self.loopPerm = copy.deepcopy(loopPerm)

       self.floatTypeA = floatTypeA
       self.floatTypeB = floatTypeB


       self.blocking = list(copy.deepcopy(blocking));
       self.vectorLength = vectorLength
       self.isBeta = isBeta
		
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

       self.matCopy = 0
       count = 0
       t=0 
       for p in perm:
	   if(p==count):
	      t=t+1
           count = count+1
       if(count == t):
	   self.matCopy = 1	

       self.rem64= 1 #Used only when perm[0] != 0
       if(self.blocking[0] == 32 and self.blocking[1] == 32):	
           self.rem64=0

       self.padding = 1 
       self.dim = len(self.perm)
       #deal with remainder
       self.remainderA = 0
       self.remainderB = 0
       if( perm[0] != 0):
           self.remainderA = size[0] % self.blocking[0]
           self.remainderB = size[perm[0]] % self.blocking[1]
       elif(self.matCopy == 0):
           self.remainderA = size[1] % self.blocking[0]
           self.remainderB = size[perm[1]] % self.blocking[1]
       else:
           if(len(self.size) != 1):
              self.remainderA = self.size[0] % self.blocking[0]
              self.remainderB = self.size[1] % self.blocking[1]
           else:
              self.remainderA = self.size[0] % (self.blocking[0]*self.blocking[1])
	

       self.remainderIndexA = -1
       self.remainderIndexB = -1
       if(self.perm[0] != 0): 
           #if(self.remainderA != 0):
           self.remainderIndexA = 0  
           #if(self.remainderB != 0):
           self.remainderIndexB = self.perm[0]
       elif(self.matCopy == 0):  
           #if(self.remainderA != 0):
           self.remainderIndexA = 1
           #if(self.remainderB != 0):
           self.remainderIndexB = self.perm[1]

           

       self.reminderIntersect = 0	
       self.remainderIntersectIndex = -1
       #if(self.remainderA !=0 and self.remainderB != 0):
       self.remainderIntersectIndex = self.remainderIndexB
       self.remainderIntersect = self.remainderB	

       #compute leading dimensions
       self.lda = copy.deepcopy(lda)
       self.ldb = copy.deepcopy(ldb)


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
       
       self.indent = "   "
       self.tiling = [32,32] #TODO this should be a function of the precision
       self.cost = 0

       if(self.vectorLength % self.tiling[0] != 0 ):
           print "[TTC] ERROR: vectorLength is not divisible by %d."%self.tiling[0]
           exit(-1)
	 


    def getCudaImplementation(self,fastestVersion=0):
	code = ""
	code += self.getCudaTransposeHeader(1,fastestVersion)
	code += "\n{\n"
	code += "//loopPerm = [ %s ]\n\n"%(self.getloopPermVersion())
	code += "   const int blockA = " + str(self.blocking[0]) + ";\n"
	code += "   const int blockB = " + str(self.blocking[1]) + ";\n"
	code += self.declareVariables()
	if(self.perm[0]!=0):
	    code +="    __shared__ %s tile[32][33];\n\n"%(self.floatTypeA)
	    code += self.getIndices(self.loopPerm, self.blocking[0],self.blocking[1])
	    code +="   const %s *Atmp = &A[%s];\n"%(self.floatTypeA,self.getOffsetA())
	    code +="   %s *Btmp = &B[%s];\n"%(self.floatTypeB,self.getOffsetB())
	    code +="   const %s *Aref = &A[%s];\n"%(self.floatTypeA,self.getOffsetA())
	    code +="   %s *Bref = &B[%s];\n"%(self.floatTypeB,self.getOffsetB())	
	    code +="   int lda_kernel = lda%d;\n"%(self.perm[0])
	    code +="   int ldb_kernel = ldb%d;\n"%(self.ldout)
	    
	    code +="   if(i%d < (size%d-remainder%d) && i%d < (size%d-remainder%d))\n"%(self.remainderIndexA, self.remainderIndexA, self.remainderIndexA ,self.remainderIndexB, self.remainderIndexB, self.remainderIndexB)
	    code +="   {"
            code += self.generateBlockedCode(self.blocking)
	    code +="   }\n"
	    code +="   else if(i%d >= (size%d-remainder%d) && i%d < (size%d-remainder%d))\n"%(self.remainderIndexA,  self.remainderIndexA ,self.remainderIndexA , self.remainderIndexB, self.remainderIndexB, self.remainderIndexB)
	    code +="   { //remainder in size%d\n"%(self.remainderIndexA)
            if(self.rem64 == 0): 
	       if(self.isBeta):
	    	   code +="       %s(Atmp,Btmp,alpha,beta,lda_kernel,ldb_kernel,remainder%d, 32,tile);\n"%(self.getRemainderKernelHeader(), self.remainderIndexA)
	       else:
	    	   code +="       %s(Atmp,Btmp,alpha,lda_kernel,ldb_kernel,remainder%d, 32,tile);\n"%(self.getRemainderKernelHeader(), self.remainderIndexA)
            else:
	       code +=self.getRemainder64()
	    code +="   }\n"
            code +="   else if(i%d >= (size%d-remainder%d) && i%d < (size%d-remainder%d))\n"%(self.remainderIndexB, self.remainderIndexB,self.remainderIndexB ,self.remainderIndexA, self.remainderIndexA, self.remainderIndexA)
	    code +="   { //remainder in size%d\n"%(self.remainderIndexB)
            if(self.rem64 == 0):
	       if(self.isBeta):
	    	    code +="       %s(Atmp,Btmp,alpha,beta,lda_kernel,ldb_kernel,32,remainder%d,tile);\n"%(self.getRemainderKernelHeader(), self.remainderIndexB)
	       else:
	    	    code +="       %s(Atmp,Btmp,alpha,lda_kernel,ldb_kernel,32,remainder%d,tile);\n"%(self.getRemainderKernelHeader(), self.remainderIndexB)
	    else:
	       code +=self.getRemainder64()
	    code +="   }\n"
	    code +="   else\n"
            if(self.rem64 == 0):
	       if(self.isBeta):
	    	   code +="       %s(Atmp,Btmp,alpha,beta,lda_kernel,ldb_kernel,remainder%d,remainder%d,tile);\n"%(self.getRemainderKernelHeader(), self.remainderIndexA, self.remainderIndexB)
	       else:
	    	   code +="       %s(Atmp,Btmp,alpha,lda_kernel,ldb_kernel,remainder%d,remainder%d,tile);\n"%(self.getRemainderKernelHeader(),self.remainderIndexA ,self.remainderIndexB)
            else:
	       code +="   {\n"
	       code +=self.getRemainder64()
	       code +="   }\n"	 
	              
	elif(self.matCopy == 0):
            code += self.getIndices(self.loopPerm, self.blocking[0],self.blocking[1])
	    code += self.getPerm0Loop()
        else:
            code += self.getIndices(self.loopPerm, self.blocking[0],self.blocking[1])
            code += self.getMatCopy()
	    
        code += "}\n"
	
	return code


    def getPerm0Loop(self):
	code = ""
	if(self.size[0] < self.vectorLength):
	    if(self.vectorLength % self.size[0] == 0):
		jump = self.vectorLength/self.size[0]
		code += "   int j0 = threadIdx.x/size0;\n"
                code += "   for(int i=0; i<blockA; i++)\n"
		code += "      for(int j=j0; j<blockB; j=j+%d)\n"%jump
		code += "      {\n"
		code += "         int i0 = threadIdx.x % size0;\n"
                #if(self.remainderA !=0 or self.remainderB !=0):
		code += "           if((i1+i*nba) < size1 && (i%d+j*nbb) < size%d)\n"%(self.perm[1],self.perm[1])
                code += self.getUpdateString("              ") 
		code +="        }\n"
	    
	    elif(self.vectorLength/self.size[0] > 1):
		jump = self.vectorLength/self.size[0]
		code += "   if(threadIdx.x < %d)\n"%(self.size[0]*jump)
		code += "   {\n"		
		code += "      int j0 = threadIdx.x/size0;\n"
                code += "      for(int i=0; i<blockA; i++)\n"
		code += "         for(int j=j0; j<blockB; j=j+%d)\n"%(jump)
		code += "         {\n"
		code += "              int i0 = threadIdx.x % size0;\n"
                #if(self.remainderA !=0 or self.remainderB !=0):
		code += "          if((i1+i*nba) < size1 && (i%d+j*nbb) < size%d)\n"%(self.perm[1],self.perm[1])
                code += self.getUpdateString("              ") 
		code += "         }\n"
	        code += "   }\n"
	         	
	    else:
                code += "   for(int i=0; i<blockA; i++)\n"
		code += "      for(int j=0; j<blockB; j++)\n"
	        code += "      {\n"
		code += "         int i0 = threadIdx.x;\n"
                #if(self.remainderA !=0 or self.remainderB !=0):
		code += "         if(i0 < size0 && (i1+i*nba) < size1 && (i%d+j*nbb) < size%d)\n"%(self.perm[1],self.perm[1])
                #else:
		#    code += "         if(i0 < size0)\n"
                code += self.getUpdateString("              ") 
		code +="       }\n"

	elif(self.size[0] > self.vectorLength):
            code += "   for(int i=0; i<blockA; i++)\n"
	    code += "      for(int j=0; j<blockB; j++)\n"
	    code += "         for(int i0=threadIdx.x; i0<size0; i0=i0+%d)\n"%self.vectorLength
	    code += "         {\n" 	
            #if(self.remainderA !=0 or self.remainderB !=0):
	    code += "             if(i0 < size0 && (i1+i*nba) < size1 && (i%d+j*nbb) < size%d)\n"%(self.perm[1],self.perm[1])
            code += self.getUpdateString("              ") 

	    code += "         }\n"
	
	else:
            code += "   for(int i=0; i<blockA; i++)\n"
	    code += "      for(int j=0; j<blockB; j++)\n"
	    code += "      {\n"
	    code += "         int i0 = threadIdx.x;\n"
            #if(self.remainderA !=0 or self.remainderB !=0):
	    code += "        if(i0 < size0 && (i1+i*nba) < size1 && (i%d+j*nbb) < size%d)\n"%(self.perm[1],self.perm[1])
            code += self.getUpdateString("              ")  	 

	    code += "      }\n"

	return code
	    		   	     	

    def getRemainderKernelHeader(self):
	code=""
	code += "cuRemainderTranspose_vec%d"%(self.vectorLength)
	return code


    def generateBlockedCode(self, blocking):
        code =""
        numBlocksA = blocking[0] / self.tiling[0]
        numBlocksB = blocking[1] / self.tiling[1]
        for i in range(numBlocksA):
            for j in range(numBlocksB):
               offsetA = ""
               offsetB = ""
               if ( i == 0):
                  if( j == 0):
                     offsetA = ""
                     offsetB = ""
                  else:
                     offsetA = " + %d * lda_kernel"%(j *self.tiling[0])
                     offsetB = " + %d"%(j *self.tiling[1])
               else:
                  if( j == 0):
                     offsetA =" + %d"%(i *self.tiling[0])
                     offsetB =" + %d * ldb_kernel"%(i *self.tiling[1])
                  else:
                     offsetA = " + %d + %d * lda_kernel"%(i * self.tiling[0],j * self.tiling[0])
                     offsetB = " + %d + %d * ldb_kernel"%(j * self.tiling[1],i * self.tiling[1])

               code +="\n     //offset\n"
               code +="      Atmp = Aref %s;\n"%(offsetA)
               code +="      Btmp = Bref %s;\n\n"%(offsetB)
               #code +="  // if(blockIdx.x < %d )\n"%(self.getNumBlocks())
               if(self.isBeta):
                   code +="      cuSharedMemTranspose_%dx%d_vec%d(Atmp,Btmp,alpha,beta,lda_kernel,ldb_kernel,tile);\n"%(self.tiling[0],self.tiling[1],self.vectorLength)
               else: 
                   code +="      cuSharedMemTranspose_%dx%d_vec%d(Atmp,Btmp,alpha,lda_kernel,ldb_kernel,tile);\n"%(self.tiling[0],self.tiling[1],self.vectorLength)

        return code




    def getSharedTransposeKernel(self):
	code = ""
        code +="static __device__\n"

        if(self.isBeta):
            code +="void cuSharedMemTranspose_%dx%d_vec%d(const %s *Atmp, %s *Btmp, const %s alpha, const %s beta, int lda, int ldb, %s tile[32][33])\n"%(self.tiling[0],self.tiling[1],self.vectorLength, self.floatTypeA, self.floatTypeB,self.alphaFloatType, self.betaFloatType, self.floatTypeA)
        else:    
            code +="void cuSharedMemTranspose_%dx%d_vec%d(const %s *Atmp, %s *Btmp, const %s alpha, int lda, int ldb, %s tile[32][33])\n"%(self.tiling[0],self.tiling[1],self.vectorLength, self.floatTypeA, self.floatTypeB, self.alphaFloatType, self.floatTypeA)
        code +="{\n\n" 
        code +="   const int TILE_DIM_X = %d;\n"%(self.tiling[0])
        code +="   const int TILE_DIM_Y = %d;\n"%(self.tiling[1])
        code +="   const int THREADS_PER_ROW = %d;\n\n"%(self.vectorLength/self.tiling[0])
        if(self.floatTypeA == "cuFloatComplex"):
            code +=   "%scuFloatComplex cuAlpha = make_cuFloatComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuFloatComplex"):
            if(self.isBeta):
                code +=   "%scuFloatComplex cuBeta = make_cuFloatComplex(beta,0.0);\n"%self.indent

        if(self.floatTypeA == "cuDoubleComplex"):
            code +=   "%scuDoubleComplex cuAlpha = make_cuDoubleComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuDoubleComplex"):
            if(self.isBeta):
                code +=   "%scuDoubleComplex cuBeta = make_cuDoubleComplex(beta,0.0);\n"%self.indent

        code +="   int id = threadIdx.x;\n"
        code +="   int rowId = id & %d;\n"%(self.tiling[1] - self.padding)
        code +="   int colId = id / TILE_DIM_X;\n\n"
        code +="   for (int j = colId; j < TILE_DIM_Y; j+= THREADS_PER_ROW)\n"
        code += self.getSharedMemoryUpdateString("       ", "toTile")
        code +="   __syncthreads();\n\n"
        code +="   for (int j = colId; j < TILE_DIM_Y; j+= THREADS_PER_ROW)\n"
        code += self.getSharedMemoryUpdateString("       ", "fromTile")

        code +="   __syncthreads();\n\n}\n\n"
	return code


    def getRemainderTransposeKernel(self, blocking):
	code = ""
        code +="static __device__\n"

        if(self.isBeta):
            code +="void %s(const %s *Atmp, %s *Btmp, const %s alpha, const %s beta, int lda, int ldb, int remainderx, int remaindery, %s tile[32][33])\n"%(self.getRemainderKernelHeader(), self.floatTypeA, self.floatTypeB,self.alphaFloatType, self.betaFloatType, self.floatTypeA)
        else:    
            code +="void %s(const %s *Atmp, %s *Btmp, const %s alpha, int lda, int ldb, int remainderx, int remaindery, %s tile[32][33])\n"%(self.getRemainderKernelHeader(), self.floatTypeA, self.floatTypeB, self.alphaFloatType, self.floatTypeA)
        code +="{\n\n" 
        code +="   const int TILE_DIM_X = %d;\n"%(blocking)
        code +="   const int TILE_DIM_Y = %d;\n"%(blocking)
        code +="   const int THREADS_PER_ROW = %d;\n\n"%(self.vectorLength/blocking)
        if(self.floatTypeA == "cuFloatComplex"):
            code +=   "%scuFloatComplex cuAlpha = make_cuFloatComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuFloatComplex"):
            if(self.isBeta):
                code +=   "%scuFloatComplex cuBeta = make_cuFloatComplex(beta,0.0);\n"%self.indent

        if(self.floatTypeA == "cuDoubleComplex"):
            code +=   "%scuDoubleComplex cuAlpha = make_cuDoubleComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuDoubleComplex"):
            if(self.isBeta):
                code +=   "%scuDoubleComplex cuBeta = make_cuDoubleComplex(beta,0.0);\n"%self.indent

        code +="   int id = threadIdx.x;\n"
        code +="   int rowId = id & %d;\n"%(blocking - self.padding)
        code +="   int colId = id / TILE_DIM_X;\n\n"
        code +="   for (int j = colId; j < TILE_DIM_Y; j+= THREADS_PER_ROW)\n"
	code +="   {\n"
	code +="      if(rowId < remainderx && j < remaindery)\n"
        code += self.getSharedMemoryUpdateString("           ", "toTile")    
	code +="      else\n"
        if(self.floatTypeA == "float" or self.floatTypeA == "double"):
	    code +="            tile[j][rowId] = 0;\n"
        if(self.floatTypeA == "cuFloatComplex"):
	    code +="            tile[j][rowId] = make_cuFloatComplex(0.0,0.0);\n"
        if(self.floatTypeA == "cuDoubleComplex"):
	    code +="            tile[j][rowId] = make_cuDoubleComplex(0.0,0.0);\n"
	code +="    }\n"  
        code +="   __syncthreads();\n\n"
        code +="   for (int j = colId; j < TILE_DIM_Y; j+= THREADS_PER_ROW)\n"
	code +="   {\n"
	code +="      if(rowId < remaindery && j < remainderx)\n"
        code += self.getSharedMemoryUpdateString("           ","fromTile")

	code +="    }\n"
        code +="   __syncthreads();\n\n}\n\n"

	return code

  


    def getRemainder64(self):
	code = "\n"
        code +="      int rowId = threadIdx.x&63;\n"
        code +="      int colId = threadIdx.x/64;\n"
	code +="      for(int j=colId; j<64; j=j+%d)\n"%(self.vectorLength/64)
        #if(self.remainderA != 0 or self.remainderB !=0):
        code +="       if((i%d+rowId)<size%d && (i%d+j)<size%d)\n"%(0,0, self.perm[0], self.perm[0])
        code += self.getUpdateString("           ") 

	return code    
        

      



    def getHostCall(self,fastestVersion=0): 
	code = "\n\n"
	code += self.getCudaTransposeHeader(0,fastestVersion)
	code += "\n{\n"

	code += "//   int numBlocks = %d;\n"%(self.getNumBlocks()[0])
        code += self.getNumBlocks()[1]
        code += "\n\n"
	code += "    int *d_size, *d_lda, *d_ldb;\n"
	code += "    cudaMalloc(&d_size,%d*sizeof(int));\n"%(self.dim) #TODO this should become just a single malloc
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
	code += "    cudaMalloc(&d_lda,%d*sizeof(int));\n"%(self.dim)
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
	code += "    cudaMalloc(&d_ldb,%d*sizeof(int));\n"%(self.dim)
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
	code += "    cudaMemcpy(d_size, size,%d*sizeof(int), cudaMemcpyHostToDevice);\n"%(self.dim)
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
	code += "    cudaMemcpy(d_lda, lda,%d*sizeof(int), cudaMemcpyHostToDevice);\n"%(self.dim)
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
	code += "    cudaMemcpy(d_ldb, ldb,%d*sizeof(int), cudaMemcpyHostToDevice);\n"%(self.dim)
        code +=  ttc_util.getCudaErrorChecking("    ", "hostCall")
        code += "\n\n"

	if(self.isBeta):
	    code += "   %s<<<numBlocks,%d>>>(A,B,alpha,beta,d_size,d_lda,d_ldb);\n"%(self.getHeaderName(1, fastestVersion), self.vectorLength)
	else:
	    code += "   %s<<<numBlocks,%d>>>(A,B,alpha,d_size,d_lda,d_ldb);\n"%(self.getHeaderName(1, fastestVersion), self.vectorLength)
	code += "   cudaDeviceSynchronize();\n\n"
        code +=  ttc_util.getCudaErrorChecking("   ", self.getHeaderName(1, fastestVersion))
	code += "   cudaFree(d_size);\n"
	code += "   cudaFree(d_lda);\n"
	code += "   cudaFree(d_ldb);\n"
        code += "}\n" 

	return code

    def getSizeStr(self):
	code = ""
	for s in self.size:
	   code += " %d "%(s)
	return code
	 

    def getCudaTransposeHeader(self,device=1,fastestVersion=0):
        code = ""
        if(device):
            functionType = "__global__\nvoid "
        else:
            functionType = "extern \"C\" void "
 
	
        if(self.isBeta):
           code +="%s %s( const %s *A, %s *B, const %s alpha, const %s beta, const int *size, const int *lda, const int *ldb)"%(functionType, self.getHeaderName(device,fastestVersion), self.floatTypeA,self.floatTypeB,self.alphaFloatType,self.betaFloatType)
        else:
           code +="%s %s( const %s *A, %s *B, const %s alpha, const int *size, const int *lda, const int *ldb)"%(functionType, self.getHeaderName(device, fastestVersion), self.floatTypeA,self.floatTypeB,self.alphaFloatType)
        		
	return code


    def getVersionName(self):
        versionName = ""
        versionName += "v"
        found0 = 0
        for i in self.loopPerm:
            if(i == 0):
               found0 = 1
            versionName += str(i)
        if(self.perm[0] == 0 and not found0):
            versionName += str(0) #0 is always the innermost loop in this case

        if(len(self.size) != 1):
           versionName += "_%dx%d_vec%d"%(self.blocking[0], self.blocking[1], self.vectorLength)
        else:
           versionName += "_1024x1_vec%d"%(self.vectorLength)

        return versionName


    def getHeaderName(self, device=1, fastestVersion=0):

        
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
            
        if(device):
            transposeName = "%sCuKernel_"%tmpChar
	else:
            transposeName = "%sCuTranspose_"%tmpChar

        for i in self.perm:
            transposeName += str(i)
  
	transposeName +="_"
        for idx in range(len(self.size)):
             transposeName += "%d"%(self.size[idx])
             if(idx != len(self.size)-1):
		 transposeName +="x"

        if( not self.isBeta ):
           transposeName +="_bz"

        if(len(self.size) != 1):
	   if(device):
	       code = "%s_v%s_%dx%d_vec%d"%(transposeName,self.getloopPermVersion(),self.blocking[0],self.blocking[1], self.vectorLength)
	   else:
	       code = "%s_v%s_%dx%d_vec%d"%(transposeName,self.getloopPermVersion(), self.blocking[0],self.blocking[1], self.vectorLength)

        if(len(self.size) == 1):
	   if(device):
	       code = "%s_Copy_vec%d"%(transposeName, self.vectorLength)
	   else:
	       code = "%s_Copy_vec%d"%(transposeName, self.vectorLength)

        if(fastestVersion == 0):
	   return code
        else:
           return transposeName


    def getloopPermVersion(self):
	code =""
	for i in self.loopPerm:
		code += str(i)
	return code

    def getPermVersion(self):
	code =""
	for i in self.perm:
		code += str(i)
	return code
	

    def declareVariables(self):
	code=""
        for i in range(self.dim):
            code +=  "%sconst int size%d = size[%d];\n"%(self.indent,i,i)
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
        
        if(self.perm[0] != 0):
            code +=   "%sconst int remainder%d = size%d %% %d;\n"%(self.indent,self.remainderIndexA, self.remainderIndexA, self.blocking[0])
        if(self.perm[0] != 0):
            code +=   "%sconst int remainder%d = size%d %% %d;\n"%(self.indent,self.remainderIndexB, self.remainderIndexB, self.blocking[1])


        if(self.floatTypeA == "cuFloatComplex"):
            code +=   "%scuFloatComplex cuAlpha = make_cuFloatComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuFloatComplex"):
            if(self.isBeta):
                code +=   "%scuFloatComplex cuBeta = make_cuFloatComplex(beta,0.0);\n"%self.indent

        if(self.floatTypeA == "cuDoubleComplex"):
            code +=   "%scuDoubleComplex cuAlpha = make_cuDoubleComplex(alpha,0.0);\n"%self.indent
        if(self.floatTypeB == "cuDoubleComplex"):
            if(self.isBeta):
                code +=   "%scuDoubleComplex cuBeta = make_cuDoubleComplex(beta,0.0);\n"%self.indent

	return code


       

    def getNumBlocks(self):	
	size = copy.deepcopy(self.size)
        equation = ""
        
	if(self.perm[0] != 0 ):         
	    numBlocks = ((size[0]  + self.blocking[0] - 1)/self.blocking[0])*((size[self.perm[0]]+self.blocking[1]-1)/self.blocking[1])
            equation += "   int numBlocks = ((size[0] + %d -1)/%d)*((size[%d] + %d -1)/%d);\n"%(self.blocking[0],self.blocking[0],self.perm[0],self.blocking[1],self.blocking[1])
	    for i in range(1,self.dim):
		if( i != self.perm[0]):
	            numBlocks = numBlocks*size[i]
                    equation += "   numBlocks *= size[%d];\n"%i
	elif(self.matCopy == 0):        
	    numBlocks = ((size[1]  + self.blocking[0] - 1)/self.blocking[0])*((size[self.perm[1]]+self.blocking[1]-1)/self.blocking[1])
            equation += "   int numBlocks = ((size[1] + %d -1)/%d)*((size[%d] + %d -1)/%d);\n"%(self.blocking[0],self.blocking[0],self.perm[1],self.blocking[1],self.blocking[1])
	    for i in range(2,self.dim):
		if( i != self.perm[1]):
	            numBlocks = numBlocks*size[i]
                    equation += "   numBlocks *= size[%d];\n"%i
	else:
            if(len(self.size) != 1):
	       numBlocks = ((size[0]  + self.blocking[0] - 1)/self.blocking[0])*((size[1]+self.blocking[1]-1)/self.blocking[1])
               equation += "   int numBlocks = ((size[0] + %d -1)/%d)*((size[1] + %d -1)/%d);\n"%(self.blocking[0],self.blocking[0],self.blocking[1],self.blocking[1])
	       for i in range(2,self.dim):
	            numBlocks = numBlocks*size[i]
                    equation += "   numBlocks *= size[%d];\n"%i
	    else:
	       numBlocks = ((size[0]  + self.blocking[0]*self.blocking[1] - 1)/(self.blocking[0]*self.blocking[1]))
               equation += "   int numBlocks = ((size[0] + %d -1)/%d);\n"%(self.blocking[0]*self.blocking[1],self.blocking[0]*self.blocking[1])
	    	
	return (numBlocks, equation)
            

    def getOffsetA(self):
        offset = ""
        if(self.perm[0] != 0): 
            for i in range(self.dim):
                offset += "i" + str(i)
                if(self.lda[i] != 1):
                    offset += "*lda" + str(i)
                if( i != self.dim-1):
                    offset += " + "
	elif(self.matCopy == 0):
            for i in range(1,self.dim):
                if(i ==1):
		    offset += "(i" + str(i) + " + i*nba)"
		elif(i==self.perm[1]):
		    offset += "(i" + str(i) + " + j*nbb)"
		else:
                    offset += "i" + str(i)
                if(self.lda[i] != 1):
                    offset += "*lda" + str(i)
                if( i != self.dim-1):
                    offset += " + "
	else:
            if(len(self.size) != 1):
	      offset +="(i0+rowId) + (i1+j)*lda1"
              for i in range(2,self.dim):
                offset += " + "
                offset += "i" + str(i)
                if(self.lda[i] != 1):
                    offset += "*lda" + str(i)
            else:
              offset +="(i0 +j+rowId)"

        return offset


    def getOffsetB(self):
        offset = ""
	if(self.perm[0] !=0 ):
            for i in range(0,self.dim):
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
	elif(self.matCopy == 0):
            for i in range(1,self.dim):
                #find idx idxPerm
                invIdx = -1
                for j in range(self.dim):
                    if self.perm[j] == i:
                        invIdx = j
                if(i==1):
		    offset += "(i" + str(i) + " + i*nba)"
		elif(i==self.perm[1]):
		    offset += "(i" + str(i) + " + j*nbb)"
		else:
                    offset += "i" + str(i)
                if(self.ldb[invIdx] != 1):
                    offset += "*ldb" + str(invIdx)
                if( i != self.dim-1):
                    offset += " + "
	else:
            if(len(self.size) != 1): 
	      offset +="(i0+rowId) + (i1+j)*ldb1"
              for i in range(2,self.dim):
                offset += " + i" + str(i)
                if(self.lda[i] != 1):
                    offset += "*ldb" + str(i)
            else:
               offset +="(i0+j+rowId)" 

        return offset


    def getRem64OffsetA(self):
        offset = ""
        if(self.perm[0] != 0): 
            for i in range(self.dim):
                if(i==0):
                   offset +="(i0+rowId)"
                elif(i == self.perm[0]):
		   offset +="(i%d+j)"%self.perm[0]
                else:
                   offset += "i" + str(i)
                if(self.lda[i] != 1):
                    offset += "*lda" + str(i)
                if( i != self.dim-1):
                    offset += " + "

        return offset


    def getRem64OffsetB(self):
        offset = ""
	if(self.perm[0] !=0 ):
            for i in range(0,self.dim):
                #find idx idxPerm
                invIdx = -1
                for j in range(self.dim):
                    if self.perm[j] == i:
                        invIdx = j
                if(i==0):
                   offset +="(i0+rowId)"
                elif(i == self.perm[0]):
		   offset +="(i%d+j)"%self.perm[0]
                else:
                   offset += "i" + str(i)
                if(self.ldb[invIdx] != 1):
                    offset += "*ldb" + str(invIdx)
                if( i != self.dim-1):
                    offset += " + "


        return offset





    def getIndices(self,loopPerm, blockA, blockB):
       code = "   int idx = blockIdx.x;\n"
 
       if(len(self.size) != 1):
          for l in range(len(loopPerm)):
             loop = loopPerm[l]
             if(self.perm[0] != 0 or self.matCopy == 1): 
	        if(loop ==0 ):
                   if( l != len(loopPerm)-1 ):
               	      code += "   int i%d = (idx %% ((size%d+blockA-1)/blockA))*blockA;\n"%(loop,loop)
            	      code += "   idx /= ((size%d+blockA-1)/blockA);\n"%(loop)
                   else:
                      code += "   int i%d = idx*blockA;\n\n"%(loop)
                elif(loop == self.perm[0] or (self.matCopy == 1 and loop == 1)):
                   if( l != len(loopPerm)-1 ):
               	      code += "   int i%d = (idx %% ((size%d+blockB-1)/blockB))*blockB;\n"%(loop,loop)
            	      code += "   idx /= ((size%d+blockB-1)/blockB);\n"%(loop) 
                   else: 
                      code += "   int i%d = idx*blockB;\n\n"%(loop)
                else:
                   if( l != len(loopPerm)-1 ):
                      code += "   int i%d = idx %% size%d;\n"%(loop,loop)
            	      code += "   idx /= size%d;\n\n"%(loop)
                   else:   
                      code += "   int i%d = idx %% size%d;\n"%(loop,loop)
             else:
                if(loop != 0):
	           if(loop ==1 ):
               	      code += "   int i%d = idx %% ((size%d + blockA -1)/blockA);\n"%(loop,loop)
            	      code += "   idx /= ((size%d + blockA -1)/blockA);\n"%(loop)
		      code +="    int nba = (size%d + blockA -1)/blockA;\n\n"%loop	
                   elif(loop == self.perm[1]):
               	      code += "   int i%d = idx %% ((size%d + blockB -1)/blockB);\n"%(loop,loop)
            	      code += "   idx /= ((size%d+blockB-1)/blockB);\n"%(loop) 
		      code +="    int nbb = (size%d + blockB -1)/blockB;\n\n"%loop
                   else:
                      code += "   int i%d = idx %% size%d;\n"%(loop,loop)
            	      code += "   idx /= size%d;\n\n"%(loop)

       if(len(self.size) == 1):
           code +="    int i0 = (idx % ((size0 + (blockA*blockB) -1)/(blockA*blockB)))*(blockA*blockB);\n"

       return code

    def getMatCopy(self):
        if(len(self.size) != 1):
	   code = "\n"
           code +="   int rowId = threadIdx.x&%d;\n"%(self.blocking[0]-1)
           code +="   int colId = threadIdx.x/%d;\n"%(self.blocking[1])
	   code +="   for(int j=colId; j<%d; j=j+%d)\n"%(self.blocking[1],(self.vectorLength/self.blocking[0]))

          # if(self.remainderA != 0 or self.remainderB !=0):
           code +="    if((i0+rowId)<size0 && (i1+j)<size1)\n" 
           code += self.getUpdateString("        ")
	else: 
	   code = "\n"
           code +="   int rowId = threadIdx.x;\n"
           code +="   for(int j=0; j<%d; j=j+%d)\n"%(self.blocking[0]*self.blocking[1], (self.vectorLength))
           #if(self.remainderA != 0):
           code +="    if((i0+j+rowId)<size0)\n"
           code += self.getUpdateString("        ") 

	return code

    def getUpdateString(self, indent):
      code = ""
      if(self.floatTypeA == "float" or self.floatTypeA == "double"):
          if(self.perm[0] == 0 and self.matCopy == 0):
	        if(self.isBeta):
		    code += indent + "B[i0+ %s] = alpha*A[i0 + %s] + beta*B[i0 + %s];\n"%(self.getOffsetB(),self.getOffsetA(),self.getOffsetB())
	        else:
		    code += indent + "B[i0 + %s] = alpha*A[i0 + %s];\n"%(self.getOffsetB(),self.getOffsetA())
          elif(self.perm[0] != 0 and self.rem64 != 0):
	        if(self.isBeta):
                    code +="           B[%s] = alpha*A[%s] + beta*B[%s];\n"%(self.getRem64OffsetB(),self.getRem64OffsetA(),self.getRem64OffsetB())
                else:
                    code +="           B[%s] = alpha*A[%s];\n"%(self.getRem64OffsetB(), self.getRem64OffsetA()) 
          else: 
               if(self.isBeta):
                    code += indent + "B[%s] = alpha*A[%s] + beta*B[%s];\n"%(self.getOffsetB(),self.getOffsetA(),self.getOffsetB())
               else:
                    code += indent + "B[%s] = alpha*A[%s];\n"%(self.getOffsetB(), self.getOffsetA())

      if(self.floatTypeA == "cuFloatComplex"):
          CuMulA = "cuCmulf"
      if(self.floatTypeA == "cuDoubleComplex"):
          CuMulA = "cuCmul"  
      if(self.floatTypeB == "cuFloatComplex"):
          CuMulB = "cuCmulf"
      if(self.floatTypeB == "cuDoubleComplex"):
          CuMulB = "cuCmul" 
      mixed=1
      if(self.floatTypeA == self.floatTypeB):
          mixed=0
      

      if(self.floatTypeB == "cuFloatComplex" and mixed ==0):
          if(self.perm[0] == 0 and self.matCopy == 0):
	        if(self.isBeta):
		    code += indent + "B[i0+ %s] = cuCaddf(%s(cuAlpha,A[i0 + %s]) , %s(cuBeta,B[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
	        else:
		    code += indent + "B[i0 + %s] = %s(cuAlpha,A[i0 + %s]);\n"%(self.getOffsetB(),CuMulA,self.getOffsetA())
          elif(self.perm[0] != 0 and self.rem64 != 0):
	        if(self.isBeta):
                    code += indent + "B[%s] = cuCaddf(%s(cuAlpha,A[%s]) , %s(cuBeta,B[%s]));\n"%(self.getRem64OffsetB(),CuMulA,self.getRem64OffsetA(),CuMulB,self.getRem64OffsetB())
                else:
                    code +=indent + "B[%s] = %s(cuAlpha,A[%s]);\n"%(self.getRem64OffsetB(),CuMulA, self.getRem64OffsetA()) 
          else: 
               if(self.isBeta):
                    code += indent + "B[%s] = cuCaddf(%s(cuAlpha,A[%s]) , %s(cuBeta,B[%s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
               else:
                    code += indent + "B[%s] = %s(cuAlpha,A[%s]);\n"%(self.getOffsetB(),CuMulA, self.getOffsetA())
      if(self.floatTypeB == "cuDoubleComplex" and mixed ==0):
          if(self.perm[0] == 0 and self.matCopy == 0):
	        if(self.isBeta):
		    code += indent + "B[i0+ %s] = cuCadd(%s(cuAlpha,A[i0 + %s]) , %s(cuBeta,B[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
	        else:
		    code += indent + "B[i0 + %s] = %s(cuAlpha,A[i0 + %s]);\n"%(self.getOffsetB(),CuMulA,self.getOffsetA())
          elif(self.perm[0] != 0 and self.rem64 != 0):
	        if(self.isBeta):
                    code += indent + "B[%s] = cuCadd(%s(cuAlpha,A[%s]) , %s(cuBeta,B[%s]));\n"%(self.getRem64OffsetB(),CuMulA,self.getRem64OffsetA(),CuMulB,self.getRem64OffsetB())
                else:
                    code +=indent + "B[%s] = %s(cuAlpha,A[%s]);\n"%(self.getRem64OffsetB(),CuMulA, self.getRem64OffsetA()) 
          else: 
               if(self.isBeta):
                    code += indent + "B[%s] = cuCadd(%s(cuAlpha,A[%s]) , %s(cuBeta,B[%s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
               else:
                    code += indent + "B[%s] = %s(cuAlpha,A[%s]);\n"%(self.getOffsetB(),CuMulA, self.getOffsetA())

      if(self.floatTypeB == "cuFloatComplex" and mixed ==1):
          if(self.perm[0] == 0 and self.matCopy == 0):
	        if(self.isBeta):
		    code += indent + "B[i0+ %s] = cuCaddf(cuComplexDoubleToFloat(%s(cuAlpha,A[i0 + %s])) , %s(cuBeta,B[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
	        else:
		    code += indent + "B[i0 + %s] = cuComplexDoubleToFloat(%s(cuAlpha,A[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA())
          elif(self.perm[0] != 0 and self.rem64 != 0):
	        if(self.isBeta):
                    code += indent + "B[%s] = cuCaddf(cuComplexDoubleToFloat(%s(cuAlpha,A[%s])) , %s(cuBeta,B[%s]));\n"%(self.getRem64OffsetB(),CuMulA,self.getRem64OffsetA(),CuMulB,self.getRem64OffsetB())
                else:
                    code +=indent + "B[%s] = cuComplexDoubleToFloat(%s(cuAlpha,A[%s]));\n"%(self.getRem64OffsetB(),CuMulA, self.getRem64OffsetA()) 
          else: 
               if(self.isBeta):
                    code += indent + "B[%s] = cuCaddf(cuComplexDoubleToFloat(%s(cuAlpha,A[%s])) , %s(cuBeta,B[%s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
               else:
                    code += indent + "B[%s] = cuComplexDoubleToFloat(%s(cuAlpha,A[%s]));\n"%(self.getOffsetB(),CuMulA, self.getOffsetA())

      if(self.floatTypeB == "cuDoubleComplex" and mixed ==1):
          if(self.perm[0] == 0 and self.matCopy == 0):
	        if(self.isBeta):
		    code += indent + "B[i0+ %s] = cuCadd(cuComplexFloatToDouble(%s(cuAlpha,A[i0 + %s])) , %s(cuBeta,B[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
	        else:
		    code += indent + "B[i0 + %s] = cuComplexFloatToDouble(%s(cuAlpha,A[i0 + %s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA())
          elif(self.perm[0] != 0 and self.rem64 != 0):
	        if(self.isBeta):
                    code += indent + "B[%s] = cuCadd(cuComplexFloatToDouble(%s(cuAlpha,A[%s])) , %s(cuBeta,B[%s]));\n"%(self.getRem64OffsetB(),CuMulA,self.getRem64OffsetA(),CuMulB,self.getRem64OffsetB())
                else:
                    code +=indent + "B[%s] = cuComplexFloatToDouble(%s(cuAlpha,A[%s]));\n"%(self.getRem64OffsetB(),CuMulA, self.getRem64OffsetA()) 
          else: 
               if(self.isBeta):
                    code += indent + "B[%s] = cuCadd(cuComplexFloatToDouble(%s(cuAlpha,A[%s])) , %s(cuBeta,B[%s]));\n"%(self.getOffsetB(),CuMulA,self.getOffsetA(),CuMulB,self.getOffsetB())
               else:
                    code += indent + "B[%s] = cuComplexFloatToDouble(%s(cuAlpha,A[%s]));\n"%(self.getOffsetB(),CuMulA, self.getOffsetA())

      return code


    def getSharedMemoryUpdateString(self,indent,direction):
      code = ""
      if(self.floatTypeA == "float" or self.floatTypeA == "double"):
	  if(direction == "toTile"):
            code += indent + "tile[j][rowId] =  alpha * Atmp[rowId + j * lda];\n"
          if(direction == "fromTile"):  
            if(self.isBeta):  
                 code += indent + "Btmp[j * ldb + rowId] = tile[rowId][j] + beta*Btmp[j * ldb + rowId];\n"
            else:
                 code += indent + "Btmp[j * ldb + rowId] = tile[rowId][j];\n"

      if(self.floatTypeA == "cuFloatComplex"):
          CuMulA = "cuCmulf"
      if(self.floatTypeA == "cuDoubleComplex"):
          CuMulA = "cuCmul"  
      if(self.floatTypeB == "cuFloatComplex"):
          CuMulB = "cuCmulf"
      if(self.floatTypeB == "cuDoubleComplex"):
          CuMulB = "cuCmul" 
      mixed=1
      if(self.floatTypeA == self.floatTypeB):
          mixed=0

      if(self.floatTypeB == "cuFloatComplex" and mixed == 0):
	  if(direction == "toTile"):
            code += indent + "tile[j][rowId] =  %s(cuAlpha , Atmp[rowId + j * lda]);\n"%CuMulA
          if(direction == "fromTile"):  
            if(self.isBeta):  
                 code += indent + "Btmp[j * ldb + rowId] = cuCaddf(tile[rowId][j] , %s(cuBeta,Btmp[j * ldb + rowId]));\n"%CuMulB
            else:
                 code += indent + "Btmp[j * ldb + rowId] = tile[rowId][j];\n"
      if(self.floatTypeB == "cuDoubleComplex" and mixed == 0):
	  if(direction == "toTile"):
            code += indent + "tile[j][rowId] =  %s(cuAlpha , Atmp[rowId + j * lda]);\n"%CuMulA
          if(direction == "fromTile"):  
            if(self.isBeta):  
                 code += indent + "Btmp[j * ldb + rowId] = cuCadd(tile[rowId][j] , %s(cuBeta,Btmp[j * ldb + rowId]));\n"%CuMulB
            else:
                 code += indent + "Btmp[j * ldb + rowId] = tile[rowId][j];\n"

      if(self.floatTypeB == "cuFloatComplex" and mixed == 1):
	  if(direction == "toTile"):
            code += indent + "tile[j][rowId] =  %s(cuAlpha , Atmp[rowId + j * lda]);\n"%CuMulA
          if(direction == "fromTile"):  
            if(self.isBeta):  
                 code += indent + "Btmp[j * ldb + rowId] = cuCaddf(cuComplexDoubleToFloat(tile[rowId][j]) , %s(cuBeta,Btmp[j * ldb + rowId]));\n"%CuMulB
            else:
                 code += indent + "Btmp[j * ldb + rowId] = cuComplexDoubleToFloat(tile[rowId][j]);\n"

      if(self.floatTypeB == "cuDoubleComplex" and mixed == 1):
	  if(direction == "toTile"):
            code += indent + "tile[j][rowId] =  %s(cuAlpha , Atmp[rowId + j * lda]);\n"%CuMulA
          if(direction == "fromTile"):  
            if(self.isBeta):  
                 code += indent + "Btmp[j * ldb + rowId] = cuCadd(cuComplexFloatToDouble(tile[rowId][j]) , %s(cuBeta,Btmp[j * ldb + rowId]));\n"%CuMulB
            else:
                 code += indent + "Btmp[j * ldb + rowId] = cuComplexFloatToDouble(tile[rowId][j]);\n"

      return code



    #def getCostLoop(self):
    #    self.cost = ttc_util.getCostLoop(self.loopPerm, self.perm, self.size)
    #    return self.cost








   


       

