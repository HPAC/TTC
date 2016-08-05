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
import ttc
from ttc import ttc_util


failed = []

def test_main():

    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

    print "Usage: python ./%s [-v : Verbose output, -q : quick, --compiler=[g++,icpc]]\n"%sys.argv[0]

    _compiler = "icpc"
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
    alphaValues = [1.1]
    betaValues = [ 1, -1, 1.2, 0]
    permutations = [[2,0,1],[1,0],[0,2,1],[1,2,0],[1,0,2],[2,1,0],[3,1,0,2]] 
    sizes = [
                [[11,8,3],[16,4,25]],
                [[7,3],[7,8],[8,7],[8,8],[16,11],[11,16],[17,17],[17,23]],
                [[11,8,3],[1,4,3],[11,7,17],[11,17,7],[7,32,32]],
                [[11,8,3],[4,4,3],[12,8,17],[33,17,7],[32,32,4]],
                [[11,8,3],[16,16,3],[12,8,17],[33,17,7],[32,32,4]],
                [[11,8,3],[16,4,25],[12,8,17],[33,17,9],[32,32,4]],
                [[17,3,7,9]]
            ]
    def runTest(perm, size, alpha, beta, numImplementations, floatType, parallelize,
            prefetchDistances, architecture, streamingStores = 0):
        permStr = ""
        for p in perm:
            permStr += str(p)+","
        sizeStr = ""
        for s in size:
            sizeStr += str(s)+","
        
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
        # setup arguments
        ###########################################
        ttc_args = ttc_util.TTCargs(perm, size)
        ttc_args.alpha = alpha
        ttc_args.beta = beta
        ttc_args.affinity = ""
        ttc_args.numThreads = _numThreads
        ttc_args.floatTypeA = floatParamA
        ttc_args.floatTypeB = floatParamB
        ttc_args.streamingStores = streamingStores
        ttc_args.maxNumImplementations = numImplementations
        ttc_args.ignoreDatabase = 1
        ttc_args.lda = []
        ttc_args.ldb = []
        ttc_args.debug = 0
        ttc_args.architecture = architecture
        ttc_args.align = 1
        ttc_args.blockings = []
        ttc_args.loopPermutations = []
        ttc_args.prefetchDistances  = []
        ttc_args.scalar = 0
        ttc_args.silent = 1
        ttc_args.hotA = 0
        ttc_args.hotB = 0 
 

        ###########################################
        # run test
        ###########################################
        try:
            (transposeName, bandwidth) = ttc.ttc.generateTransposition( ttc_args )
        except:
          print FAIL + "[Error] Test failed: ", ENDC
          print ttc_args.getCommandLineString()
          return 1


        print OKGREEN + "[Success] ",ttc_args.getCommandLineString(), ENDC
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
                        numImplementations = 40
                        failCount += runTest(perm, size, alpha, beta, numImplementations,
                                floatType, parallelize, prefetchDistances, architecture )

                        if( _quick ):
                            break

    alpha = 1.1
    beta = 0.0
#test all permutations and sizs
    for streamingStores in [0,1]:
        for c in range(len(sizes)):
            for floatType in floatTypes:
                perm = permutations[c]
                for size in sizes[c]:
                    count += 1
                    numImplementations = 40
                    failCount += runTest(perm, size, alpha, beta, numImplementations, floatType, parallelize, prefetchDistances, architecture, streamingStores  )
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
                numImplementations = 40
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
