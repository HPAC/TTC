#!/usr/bin/env python

### __VERSION__ 40
#
# This script tests the correctness of the tensor contraction compiler 
# for multi-dimensional transpositions.
#
# Copyright (C) 2015 Paul Springer (springer@aices.rwth-aachen.de)
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

import sys
import os
import subprocess
import shutil 
from ttc import transposeGenerator
from ttc import ttc_util


failed = []

def test_main():

    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

    print "Usage: python ./%s [-v : Verbose output, -q : quick, --compiler=[intel,gcc]]\n"%sys.argv[0]

    _compiler = "intel"
    _allowedArguments = ["-v","-q"]
    _quick = 0
    _numThreads = 10
    _papi = 0
    _noTest = 0
    _scalar = 0
    _align = 1
    _architectures = ["avx", "knc"]

    for arg in sys.argv:
        if( arg == sys.argv[0]): continue
        valid = 0
        for allowed in _allowedArguments:
            if( arg == allowed ):
                valid = 1
                break
        if valid == 0:
            print "ERROR: argument %s not valid."%arg

        if arg.find("--compiler=") != -1:
            _compiler = arg.split("=")[1]
        if arg == "-q":
            _quick = 1

    compiler_version = ttc_util.getCompilerVersion(_compiler)
    print "Using following compiler: %s"%(compiler_version)


    debug = 0
    if len(sys.argv) == 2 and sys.argv[1] == "-v":
        debug = 1

#make a copy of tcc

    nRepeat = "1"
    DEVNULL = open(os.devnull, 'wb')

    floatTypes = ["d","s","c","z","sd","ds","cz","zc" ]
    alphaValues = [ 1, -1, 1.1]
    betaValues = [ 1, -1, 1.2, 0]
    permutations = [[2,0,1],[1,0],[0,2,1],[1,2,0],[1,0,2],[2,1,0],[3,1,0,2]] 
    sizes = [
                [[11,8,3],[16,4,25]],
                [[7,7],[7,8],[8,7],[8,8],[16,11],[11,16],[17,17],[17,23]],
                [[11,8,3],[1,4,3],[11,7,17],[11,17,7],[7,32,32]],
                [[11,8,3],[4,4,3],[12,8,17],[33,17,7],[32,32,4]],
                [[11,8,3],[16,16,3],[12,8,17],[33,17,7],[32,32,4]],
                [[11,8,3],[16,4,25],[12,8,17],[33,17,9],[32,32,4]],
                [[17,3,7,9]]
            ]
    _loopOrder=[]
    _blockings=[]
    _mpi = 0

    def runTest(perm, size, alpha, beta, numImplementations, floatType, parallelize,
            prefetchDistances, architecture):
        lda =[1] 
        for i in range(1,len(size)):
            lda.append(lda[-1] * size[i-1])
        ldb = [1]
        for i in range(1,len(size)):
            ldb.append(ldb[i-1]*size[perm[i-1]])
        permStr = ""
        for p in perm:
            permStr += str(p)+","
        sizeStr = ""
        for s in size:
            sizeStr += str(s)+","
        
        version = "--alpha=%f --beta=%f --perm=%s --size=%s --dataType=%s"%(alpha,beta,permStr[:-1],sizeStr[:-1], floatType)
        print "Current version:", version
 
        floatParamA = "float"
        if floatType[0] == "d":
            floatParamA = "double"
        elif floatType[0] == "z":
            floatParamA = "double complex"
        elif floatType[0] == "c":
            floatParamA = "float complex"

        if( len(floatType) == 2 ):
            floatParamB = "float"
            if floatType[1] == "d":
                floatParamB = "double"
            elif floatType[1] == "z":
                floatParamB = "double complex"
            elif floatType[1] == "c":
                floatParamB = "float complex"
        else:
            floatParamB = floatParamA

        ###########################################
        # generate versions
        ###########################################
        generator = transposeGenerator(perm, _loopOrder, size, alpha, beta, numImplementations,
                floatParamA, floatParamB, parallelize, 0, prefetchDistances,_blockings, _papi, _noTest,
                _scalar, _align, architecture, _mpi, lda, ldb, 0)
        generator.generate()

        ###########################################
        # compile versions
        ###########################################
        print "[make] Compile ...                                 "
        if debug == 0:
            ret = subprocess.call(["make", "-j", _compiler], stdout=DEVNULL, stderr=subprocess.STDOUT)
        else:
            ret = subprocess.call(["make", "-j", _compiler])
        if ret != 0 :
            print FAIL+"[Error] compilation of version ", version, " failed." + ENDC
            failed.append(version)
            return 1;

        ###########################################
        # run versions
        ###########################################
        #set environment variables
        os.environ["OMP_NUM_THREADS"] = str(_numThreads)
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

        print "[running] checking correctness"
        proc = subprocess.Popen(['./transpose.exe'],stdout=subprocess.PIPE)

        error = 0
        while True:
            line = proc.stdout.readline()
            line = line.lower()
            if( line.find("error") != -1 ):
                error = 1;
                break;
            if( line.find("maximal bandwidth") != -1 ): #use this as termination criterion
                break;
        if error != 0:
            print proc.poll()
            print FAIL+"[Error] runtime error while executing ", version, ENDC
            failed.append(version)
            return 1;

        print OKGREEN + "[Success] ",version, ENDC
        return 0
            


###########################################
# generate versions
###########################################
    failCount = 0
    count = 0

    architecture = _architectures[0]

    parallelize = 0
    prefetchDistances = [0]
#test for all combinations of alpha and beta
    for alpha in alphaValues:
        for beta in betaValues:
            for floatType in floatTypes:
                for c in range(2): #only use the first two permutations to test all combinations of alpha and beta
                    perm = permutations[c]
                    for size in sizes[c]:
                        count += 1
                        numImplementations = 200
                        failCount += runTest(perm, size, alpha, beta, numImplementations,
                                floatType, parallelize, prefetchDistances, architecture )

                        if( _quick ):
                            break

    alpha = 1.0
    beta = 0.0
#test all permutations and sizs
    for c in range(len(sizes)):
        for floatType in floatTypes:
            perm = permutations[c]
            for size in sizes[c]:
                count += 1
                numImplementations = 200
                failCount += runTest(perm, size, alpha, beta, numImplementations, floatType, parallelize, prefetchDistances, architecture )
                if( _quick ):
                    break


    alpha = 1.0
    beta = 1.1
    parallelize = 1
#test all parallelism and prefetch distance
    distances =  range(1,3)
    for c in range(len(sizes)):
        for floatType in floatTypes:
            perm = permutations[c]
            for size in sizes[c]:
                count += 1
                numImplementations = 200
                failCount += runTest(perm, size, alpha, beta, numImplementations, floatType, parallelize, distances, architecture )
                if( _quick ):
                    break

                    

###########################################
# print result
###########################################
    if failCount == 0:
        print OKGREEN + "[Success] all %d tests passed."%count, ENDC
    else:
        print FAIL + "[Error] %d out of %d tests failed."%(failCount, count), ENDC

    for version in failed:
        print "Version: %s failed."%version


if __name__ == "__main__":
    test_main()
