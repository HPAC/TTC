def genEigen(size, perm, floatType, floatTypeSize, numThreads):
   sizeStr = ""
   sizeStrPerm = ""
   sizeBytes = floatTypeSize
   for i in range(len(size)):
       s = size[i]
       sizeStr += str(s)+","
       sizeStrPerm += str(size[perm[i]])+","
       sizeBytes *= s
   sizeStr = sizeStr[0:-1] #delete last ','
   sizeStrPerm = sizeStrPerm[0:-1] #delete last ','

   permStr = ""
   for s in perm:
       permStr += str(s)+","
   permStr = permStr[0:-1] #delete last ','


   code = "#define EIGEN_USE_THREADS\n"
   code += "#include <unsupported/Eigen/CXX11/Tensor>\n"
   code += "#include <unsupported/Eigen/CXX11/ThreadPool>\n"
   code += "#include <stdio.h>\n"
   code += "#include <stdlib.h>\n"
   code += "#include <omp.h>\n"
   code += "\n"
   code += "void trashCache(float* trash1, float* trash2, int nTotal){\n"
   code += "   for(int i = 0; i < nTotal; i ++) \n"
   code += "      trash1[i] += 0.99 * trash2[i];\n"
   code += "}\n"
   code += "void example(int argc, char** argv)\n{\n"

   code += "  Eigen::ThreadPool pool(%d);\n"%numThreads
   code += "  Eigen::ThreadPoolDevice my_device(&pool, %d);\n"%numThreads
   code += "  Eigen::Tensor<%s, %d> input(%s);\n"%(floatType,len(size),sizeStr)
   code += "  Eigen::Tensor<%s, %d> output(%s);\n"%(floatType,len(size),sizeStrPerm)
   code += "  input.setZero();\n"
   code += "  output.setZero();\n"

   code += "\n"
   code += "  float *trash1, *trash2;\n"
   code += "  int nTotal = 1024*1024*100;\n"
   code += "  trash1 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  trash2 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  //* Creates distributed tensors initialized with zeros\n"
   code += "\n"
   code += "\n"
   code += "  double minTime = 1e100;\n"
   code += "  for (int i=0; i<3; i++){\n"
   code += "     trashCache(trash1, trash2, nTotal);\n"
   code += "     double t = omp_get_wtime();\n"
   code += "     output.device(my_device) += input.shuffle(Eigen::array<int, %d>{%s});\n"%(len(perm),permStr)
   code += "     t = omp_get_wtime() - t;\n"
   code += "     minTime = (minTime < t) ? minTime : t;\n"
   code += "  }\n"
   code += "  double bytes = 3.0*%s /1024. /1024. / 1024. ; \n"%(sizeBytes)
   code += "  printf(\"%s %s %%.2lf seconds/TC, %%.2lf GiB/s\\n\",minTime, bytes/minTime);\n"%(sizeStr, permStr)
   code += " \n"
   code += "  free(trash1);\n"
   code += "  free(trash2);\n"
   code += "} \n"

   code += "\n"
   code += "int main(int argc, char ** argv){\n"
   code += "\n"
   code += "\n"
   code += "  example(argc, argv);\n"
   code += "\n"
   code += "  return 0;\n"
   code += "}\n"
   return code
 
