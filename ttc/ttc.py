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

import traceback
import multiprocessing
import sqlite3
import transposeGenerator as tg
import GPUtransposeGenerator as gputg
import ttc_util
import sql_util
import sys
import copy 
import subprocess
import itertools
import os
import shutil
import socket
import time as _time

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

DEVNULL = open(os.devnull, 'wb')

def getTransposeName( ttcArgs ):
    floatTypeA = ttcArgs.floatTypeA
    floatTypeB = ttcArgs.floatTypeB
    perm  = ttcArgs.idxPerm
    size  = ttcArgs.size
    alpha = ttcArgs.alpha 
    beta = ttcArgs.beta
    numThreads = ttcArgs.numThreads
    compiler = ttcArgs.compiler

    if(floatTypeA == "float"):
        if(floatTypeB == "float"):
            name = "s"
        else:
            name = "sd"
    if(floatTypeA == "double"):
        if(floatTypeB == "double"):
            name = "d"
        else:
            name = "ds"
    if(floatTypeA == "float complex"):
        if(floatTypeB == "float complex"):
            name = "c"
        else:
            name = "cz"
    if(floatTypeA == "double complex"):
        if(floatTypeB == "double complex"):
            name = "z"
        else:
            name = "zs"

    if(compiler == "nvcc"):
        name += "Cu"

    name += "Transpose_"

    for p in perm:
        name += str(p)

    name +="_"
    for idx in range(len(size)):
        name += "%d"%(size[idx])
        if(idx != len(size)-1):
            name +="x"

    if( compiler != "nvcc" and numThreads > 1):
        name += "_par"

    if(beta==0):
        name += "_bz"

    return name

def printEpilog(transposeName, ttcArgs):
    print OKGREEN +"[SUCCESS]" + ENDC +" Please find the generated code under ./ttc_transpositions/%s.h\n"%transposeName

    print "------------------ Usage ------------------"
    print "// 1) include header"
    print "#include \"%s.h\""%transposeName
    print ""
    print "// 2) execute transposition"
    if(ttcArgs.beta != 0):
        if(ttcArgs.architecture == "cuda"):
            print "%s(A, B, alpha, beta);"%transposeName
        else:
            print "%s<size0, size1,..., sizeN>(A, B, alpha, beta, lda, ldb);"%transposeName
    else:
        if(ttcArgs.architecture == "cuda"):
            print "%s(A, B, size, alpha);"%transposeName
        else:
            print "%s<size0, size1,..., sizeN>(A, B, alpha, lda, ldb);"%transposeName
    print "-------------------------------------------"

def printHelp():
    print "Tensor-Contraction Compiler  Copyright (C) 2015 Paul Springer\n" 
    print "This program comes with ABSOLUTELY NO WARRANTY; see LICENSE.txt for details."
    print "This is free software, and you are welcome to redistribute it"
    print "under certain conditions; see LICENSE.txt for details.\n"

    print "Usage: %s --perm=..., --size=... [optional arguments]\n"%sys.argv[0]
    print "This multi-dimensional transpose generator generates C++ code for a given transposition."
    print "The transpositions supported by the generator have the following form:\n"
    print "   B_{\Pi(i1,i2,..., iN)} = \\alpha * A_{i1,i2,..., iN} + \\beta * B_{\Pi(i1,i2,..., iN)}\n"

    print "required arguments:"
    print "   --perm=<index1>,<index2>,...,<indexN>".ljust(60), "specify the permutation"
    print "   --size=<size1>,<size2>,...,<sizeN>".ljust(60),"size of the input tensor"
    print ""
    print "optional arguments:"
    print "   --lda=<lda1>,<lda2>,...,<ldaN>".ljust(60),"leading dimension of each dimension of the input tensor"
    print "   --ldb=<ldb1>,<ldb2>,...,<ldbN>".ljust(60),"leading dimension of each dimension of the output tensor"
    print "   --maxImplementations=<value>".ljust(60),"limit the number of implementations"
    print "   ".ljust(14),"-> Default: 200; -1 denotes 'no limit'" 
    print "   --beta=<value>".ljust(60),"beta value (default: 0.0)"
    print "   --compiler=[g++,icpc,ibm,nvcc]".ljust(60),"choose compiler (default: icpc)"
    print "   --numThreads=<value>".ljust(60),"number of threads to launch"
    print """   --affinity=<text>".ljust(60),"thread affinity (default: 'granularity=fine,compact,1,0')
    The value of this command-line argument sets the value for the KMP_AFFINITY or the GOMP_CPU_AFFINITY environment variable for icpc or g++ compiler, respectively.
    For instance, using --compiler=icpc _and_ --affinity=compact,1 will set KMP_AFFINITY=compact,1.
    Similarly, using --compiler=g++ _and_ --affinity=0-4 will set GOMP_CPU_AFFINITY=0-4."""
    print """   --dataType=[s,d,c,z,sd,ds,cz,zc]".ljust(60),"select the datatype: 
    's' : single-precision (default),  
    'd' : double-precision, 
    'c' : complex, 
    'z' : doubleComplex,
    'sd', 'ds', 'cz', 'zc': mixed precision; 'xy' denotes that the input tensor and output tensor respectively use the data type 'x' and 'y'."""
    print "   --use-streamingStores".ljust(60),"enables streaming stores. Default: don't use streaming stores."
    print "   --prefetchDistances=<value>[,<value>, ...]".ljust(60),"number of blocks ahead of the current block. Default: 5"
    print "   --blockings=<value>x<value>[,<value>x<value>, ...]".ljust(60),"available blockings (default: all)"
    print "   --verbose or -v".ljust(60),"prints compiler output"
    print "   --generateOnly".ljust(60),"only generates the implementations, no timing"
    print "   --noTest".ljust(60),"no validation will be done."
    print "   --ignoreDatabase".ljust(60),"Don't use the SQL database (i.e., no lookup)."
    print "   --no-align".ljust(60),"prevents use of aligned load/store instructions (use this if your arrays are not aligned)."
    print "   --loopPerm=<index1>,<index2>,...,<indexN>[-<next permutation>]".ljust(60),"generates only the specifed loop order"
    print "   --hotA".ljust(60),"Specifying this flag will keep the input tensor A in cache while measuring (this will not have any effect if A does not fit into the cache)."
    print "   --hotB".ljust(60),"Specifying this flag will keep the input tensor B in cache while measuring (this will not have any effect if B does not fit into the cache)."
    print "   --help".ljust(60),"prints this help"
    print "   --threadsPerBlock=[128,256,512]".ljust(60),"Set the number of threads per threadblock (CUDA only). Default: 256."
    print """   --architecture=
    avx, power (experimental), avx512 (experimental), knc, cuda 
    
    Default: avx
    
    If your instruction set is not listed, you can always use the '--no-vec' option to
    disable explicit vectorization, such that TTC runs on all architectures.

    Please note: Support for Power, avx512 and KNC is still very experimental"""


    print ""
    print "Example: \"ttc --perm=1,0,2 --size=1000,768,16 --beta=1.0 --dataType=s\""
    print "This will swap indices 0 and 1 while leaving index 2 in place.\n"

def getInfoAboutVersion(version):
    tokens = version.split("_")
    loopOrderStr = tokens[0][1:]
    array = []
    for i in range(len(loopOrderStr)):
        array.append(int(loopOrderStr[i]))

    blockA = int(tokens[1].split('x')[0])
    blockB = int(tokens[1].split('x')[1])
    prefetchDistance = 0
    if( version.find("vec") != -1 ):
        prefetchDistance = int(tokens[2][3:])
    if( version.find("prefetch") != -1 ):
        prefetchDistance = int(tokens[-1])
    if(version.find("prefetch") != -1 and prefetchDistance == 0):
        print FAIL + "[TTC] ERROR: prefetch distance could not be decoded from version: ", version  + ENDC
        exit(-1)

    return (blockA, blockB, prefetchDistance, array)

def getVersion():
    f = open ("ttc.py","r")
    for line in f:
        tokens = line.split()
        if(len(tokens) == 3 and tokens[1] == "__VERSION__"):
            f.close()
            return int(tokens[2])
        else:
            f.close()
            return -1 
    f.close()
    return -1 


def createBestView(cursor, topXpercent):
    if(topXpercent < 0 or topXpercent > 100):
        print FAIL + "[TTC] ERROR: topXpercent not valid." + ENDC
        exit(-1)
    
    command = """
    CREATE VIEW IF NOT EXISTS 'fastest%dPercent' AS 
        select * from measurements join (
                select v2.* from (
                        select *,max(bandwidth) as maxBandwidth from variant group by measurement_id
                    ) v1 inner join variant v2 on 
                    v1.measurement_id=v2.measurement_id 
                where v2.bandwidth >= (%f * v1.maxBandwidth)
            ) fast on measurements.measurement_id = fast.measurement_id;
    """%(topXpercent, 1.0 - float(topXpercent)/100.)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
def createTables(cursor):

    command ="""
    CREATE TABLE IF NOT EXISTS 'size' (
      'size_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'size0' INTEGER NULL ,
      'size1' INTEGER NULL ,
      'size2' INTEGER NULL ,
      'size3' INTEGER NULL ,
      'size4' INTEGER NULL ,
      'size5' INTEGER NULL ,
      'size6' INTEGER NULL,
      'size7' INTEGER NULL,
      'size8' INTEGER NULL,
      'size9' INTEGER NULL);
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'permutation' (
      'permutation_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'idx0' INTEGER NULL ,
      'idx1' INTEGER NULL ,
      'idx2' INTEGER NULL ,
      'idx3' INTEGER NULL ,
      'idx4' INTEGER NULL ,
      'idx5' INTEGER NULL ,
      'idx6' INTEGER NULL ,
      'idx7' INTEGER NULL ,
      'idx8' INTEGER NULL ,
      'idx9' INTEGER NULL )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'measurements' (
      'measurement_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'dim' INTEGER NULL ,
      'host' VARCHAR(45) NULL ,
      'architecture' VARCHAR(45) NULL ,
      'version' VARCHAR(45) NULL ,
      'alpha' FLOAT NULL ,
      'beta' FLOAT NULL ,
      'numThreads' INTEGER NULL ,
      'compiler_version' VARCHAR(60) NULL ,
      'floatTypeA' VARCHAR(45) NULL ,
      'floatTypeB' VARCHAR(45) NULL ,
      'top1' FLOAT NULL ,
      'top5' FLOAT NULL ,
      'compilationTime' FLOAT NULL, 
      'measuringTime' FLOAT NULL, 
      'referenceBandwidth' FLOAT NULL, 
      'size_id' INTEGER NULL ,
      'permutation_id' INTEGER NULL ,
      'hotA' INTEGER NULL,
      'hotB' INTEGER NULL ,
      FOREIGN KEY ('size_id') REFERENCES 'size' ('size_id'),
      FOREIGN KEY ('permutation_id') REFERENCES 'permutation' ('permutation_id')
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'loopOrder' (
      'loopOrder_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'idx0' INTEGER NULL ,
      'idx1' INTEGER NULL ,
      'idx2' INTEGER NULL ,
      'idx3' INTEGER NULL ,
      'idx4' INTEGER NULL ,
      'idx5' INTEGER NULL ,
      'idx6' INTEGER NULL ,
      'idx7' INTEGER NULL ,
      'idx8' INTEGER NULL ,
      'idx9' INTEGER NULL 
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'variant' (
      'variant_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'blockA' INTEGER NULL ,
      'blockB' INTEGER NULL ,
      'prefetchDistance' INTEGER NULL ,
      'bandwidth' FLOAT NULL ,
      'time' FLOAT NULL ,
      'tlbMisses' FLOAT NULL ,
      'l2misses' FLOAT NULL ,
      'rankLoopOrder' INTEGER NULL ,
      'rankBlocking' INTEGER NULL ,
      'measurement_id' INTEGER NULL ,
      'loopOrder_id' INTEGER NULL ,
       FOREIGN KEY ('measurement_id') REFERENCES 'measurements' ('measurement_id'),
       FOREIGN KEY ('loopOrder_id') REFERENCES 'loopOrder' ('loopOrder_id')
       )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)

    command = """
    CREATE VIEW IF NOT EXISTS 'joined' AS 
        SELECT * FROM measurements join size on measurements.size_id =
            size.size_id join permutation on measurements.permutation_id =
            permutation.permutation_id join variant on variant.measurement_id =
            measurements.measurement_id
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)


#    #this table also has the speedup due to sw prefetching
#    command = """
#    CREATE VIEW IF NOT EXISTS 'variant_swSpeedup' AS 
#        select v1.*, (v1.bandwidth/v2.bandwidth) as prefetchSpeedup from variant v1 join 
#            ( select blockA, blockB, bandwidth, loopOrder_id, measurement_id from variant where prefetchDistance = 0) v2 on  
#                v1.measurement_id = v2.measurement_id and 
#                v1.blockA = v2.blockA and 
#                v1.blockB = v2.blockB and 
#                v1.loopOrder_id = v2.loopOrder_id
#              """
#    try:
#        cursor.execute(command)
#    except sqlite3.Error as e:
#        print FAIL + "ERROR (sql):", e.args[0], ENDC
#        exit(-1)
#
    createBestView(cursor, 0)
#    createBestView(cursor, 3)
#    createBestView(cursor, 5)
#    createBestView(cursor, 10)
#
#    #create view that has the best bandwidth for each measurement with a prefetchDistance > 0
#    command = """create view IF NOT EXISTS 'PRE' as select measurement_id, MAX(bandwidth)
#    as bandwidth ,prefetchDistance from joined WHERE prefetchDistance > 0 GROUP BY
#    measurement_id;"""
#    try:
#        cursor.execute(command)
#    except sqlite3.Error as e:
#        print FAIL + "ERROR (sql):", e.args[0], ENDC
#        exit(-1)
#
#    #create view that has the best bandwidth for each measurement with a prefetchDistance = 0
#    commmand = """create view IF NOT EXISTS 'NOPRE' as select measurement_id, MAX(bandwidth) as bandwidth from joined WHERE prefetchDistance = 0 GROUP BY measurement_id;"""
#    try:
#        cursor.execute(command)
#    except sqlite3.Error as e:
#        print FAIL + "ERROR (sql):", e.args[0], ENDC
#        exit(-1)




def getLastPrimaryKey(cursor, _logFile):
    command = "SELECT last_insert_rowid()"
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    result = cursor.fetchall() 
    return result[0][0]

def insertIntoMeasurements(cursor, dim, host, version, alpha, beta,
                        numThreads, compiler_version,
                        floatTypeA, floatTypeB,top1Speedup, top5Speedup, compilationTime, measuringTime, size_id, perm_id, _logFile, referenceBw, hotA, hotB, architecture):
    command = """INSERT INTO measurements (dim, host, architecture, version, alpha, beta,
            numThreads, compiler_version, floatTypeA, floatTypeB, top1, top5, compilationTime,
            measuringTime, referenceBandwidth, size_id, permutation_id, hotA, hotB)
                VALUES ( 
                %d, 
                '%s',
                '%s',
                %d,
                %f,
                %f,
                %d,
                '%s',
                '%s','%s', %f, %f, %f, %f, %f, %d, %d, %d, %d);"""%(dim, host, architecture, version, alpha, beta,
                        numThreads, compiler_version,
                        floatTypeA,floatTypeB,top1Speedup, top5Speedup,  compilationTime,
                        measuringTime, referenceBw, size_id, perm_id, hotA, hotB)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor, _logFile)
    return primaryKey

def insertIntoSize(cursor, size, _logFile):
    #check if this record already exists
    command = """SELECT size_id from size WHERE """
    maxDim = 10
    for i in range(maxDim):
        if( i < len(size) ):
            command += "size%d = '%d'"%(i,size[i])
        else:
            command += "size%d is NULL"%(i)
        if( i != maxDim -1 ):
            command += " and "
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    result = cursor.fetchall() 

    if( len(result) > 0 and len(result[0]) > 0):
        return result[0][0]
    else:
        #update table
        command = "INSERT INTO size( "
        for i in range(len(size)):
            command += "size%d"%i
            if( i != len(size) -1 ):
                command += ", "
        command += ") VALUES ("
        for i in range(len(size)):
            command += "%d"%size[i]
            if( i != len(size) -1 ):
                command += ", "
        command += ");"
        try:
            cursor.execute(command)
        except sqlite3.Error as e:
            print command
            print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
            traceback.print_stack()   
            exit(-1)
        primaryKey = getLastPrimaryKey(cursor, _logFile)
        return primaryKey

def insertIntoPermutation(cursor, perm, _logFile):
    #check if this record already exists
    command = """SELECT permutation_id from permutation WHERE """
    maxDim = 10
    for i in range(maxDim):
        if( i < len(perm) ):
            command += "idx%d = '%d'"%(i,perm[i])
        else:
            command += "idx%d is NULL"%(i)
        if( i != maxDim -1 ):
            command += " and "
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    result = cursor.fetchall() 

    if( len(result) > 0 and len(result[0]) > 0):
        return result[0][0]
    else:
        #update table
        command = "INSERT INTO permutation( "
        for i in range(len(perm)):
            command += "idx%d"%i
            if( i != len(perm) -1 ):
                command += ", "
        command += ") VALUES ("
        for i in range(len(perm)):
            command += "%d"%perm[i]
            if( i != len(perm) -1 ):
                command += ", "
        command += ");"
        try:
            cursor.execute(command)
        except sqlite3.Error as e:
            print command
            print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
            traceback.print_stack()   
            exit(-1)
        primaryKey = getLastPrimaryKey(cursor, _logFile)
        return primaryKey


def insertIntoLoopOrder(cursor, loopOrder, _logFile):
    #check if this record already exists
    command = """SELECT loopOrder_id from loopOrder WHERE """
    maxDim = 10
    for i in range(maxDim):
        if( i < len(loopOrder) ):
            command += "idx%d = '%d'"%(i,loopOrder[i])
        else:
            command += "idx%d is NULL"%(i)
        if( i != maxDim -1 ):
            command += " and "
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    result = cursor.fetchall() 

    if( len(result) > 0 and len(result[0]) > 0):
        return result[0][0]
    else:
        #update table
        command = "INSERT INTO loopOrder( "
        for i in range(len(loopOrder)):
            command += "idx%d"%i
            if( i != len(loopOrder) -1 ):
                command += ", "
        command += ") VALUES ("
        for i in range(len(loopOrder)):
            command += "%d"%loopOrder[i]
            if( i != len(loopOrder) -1 ):
                command += ", "
        command += ");"
        try:
            cursor.execute(command)
        except sqlite3.Error as e:
            print command
            print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
            traceback.print_stack()   
            exit(-1)
        primaryKey = getLastPrimaryKey(cursor, _logFile)
        return primaryKey

def insertIntoVariant(cursor, blockA, blockB, prefetchDistance,
                      BW, time, rankLoopOrder, rankBlocking,
                      measurement_id, loop_id, tlbMisses, l2misses, _logFile):
    command = """INSERT INTO variant( 
        blockA , 
        blockB , 
        prefetchDistance , 
        bandwidth , 
        time ,
        tlbMisses, 
        l2misses, 
        rankLoopOrder, 
        rankBlocking, 
        measurement_id,
        loopOrder_id) VALUES (%d, %d, %d, %f, %f, %f, %f, %d, %d,
        %d, %d);"""%(blockA,blockB,prefetchDistance,BW,time, tlbMisses, l2misses, rankLoopOrder, rankBlocking,measurement_id, loop_id)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "[TTC] ERROR (sql):", e.args[0], ENDC
        traceback.print_stack()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor, _logFile)
    return primaryKey

def generateTransposition( ttcArgs ):

    compiler_version = ttc_util.getCompilerVersion(ttcArgs.compiler)

    if( ttcArgs.architecture != "avx" and ttcArgs.floatTypeA != ttcArgs.floatTypeB):
        print FAIL + "[TTC] ERROR: Mixed precision is currently only supported for avx-enabled processors." + ENDC
        exit(-1)
    if( (ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512" or ttcArgs.architecture == "power") and ttcArgs.floatTypeA != "float"):
        print FAIL + "[TTC] ERROR: the selected architecture doesn't support the selected precision yet." + ENDC
        exit(-1)
    if( (ttcArgs.architecture == "avx512") ):
        print WARNING + "[TTC] WARNING: you are trying to generate code for an avx512-enabled processor. This could be either the host or a KNL coprocessor. " + ENDC

    if ttcArgs.maxNumImplementations == -1:
        ttcArgs.maxNumImplementations = 10000000

    _ttc_root = ""
    if( os.environ.has_key('TTC_ROOT') ):
        _ttc_root = os.environ['TTC_ROOT']
    else:
        print FAIL + "[TTC] ERROR: TTC_ROOT environment variable not set. Make sure that this variable points to the directory containing 'ttc.py'" + ENDC
        exit(-1)

    _ttc_root += "/ttc"
    workingDir = os.getcwd()
    os.chdir(_ttc_root)

    ttcArgs.updateDatabase = 1
    _generateOnly = 0
    _papi = 0
    _mpi = 0
    _logFile = open("log.txt","a+")
    _noTest = 0
    _database = "ttc.db"

    if(len (ttcArgs.lda) == 0):
        ttcArgs.lda = []
    if(len (ttcArgs.ldb) == 0):
        ttcArgs.ldb = []

    ###########################################
    # fuse indices
    ###########################################
    sizeOld = copy.deepcopy(ttcArgs.size)
    permOld = copy.deepcopy(ttcArgs.idxPerm)
    ldaOld = copy.deepcopy(ttcArgs.lda)
    ldbOld = copy.deepcopy(ttcArgs.ldb)
    ttc_util.fuseIndices(ttcArgs.size, ttcArgs.idxPerm, ttcArgs.loopPermutations, ttcArgs.lda, ttcArgs.ldb)
    if len(sizeOld) != len(ttcArgs.size):
        if( ttcArgs.silent != 1):
            print "Permutation and Size _before_ index-fusion: ", permOld, sizeOld, ldaOld, ldbOld
            print "Permutation and Size _after_ index-fusion: ", ttcArgs.idxPerm, ttcArgs.size, ttcArgs.lda, ttcArgs.ldb
    _dim = len(ttcArgs.size)

    ###########################################
    # set further defaults
    ###########################################

    if( ttcArgs.affinity == "" and ttcArgs.compiler != "nvcc"):
        if(ttcArgs.compiler == "g++" ):
            ttcArgs.affinity = "0-23:2 1-24:2"
            print WARNING + "WARNING: you did not specify an thread affinity. We are using: GOMP_CPU_AFFINITY=%s by default"%ttcArgs.affinity +ENDC
            print WARNING + "WARNING: The default thread affinity might be suboptimal depending on the numbering of your CPU cores. We recommend using a ''compact'' thread affinity even for g++ (i.e., simulate KMP_AFFINITY=compact)."+ENDC
        else:
            ttcArgs.affinity = "compact,1"
            print WARNING + "WARNING: you did not specify an thread affinity. We are using: KMP_AFFINITY=%s by default"%ttcArgs.affinity +ENDC
    if( ttcArgs.numThreads == 0 and ttcArgs.compiler != "nvcc"):
        print WARNING + "WARNING: you did not specify the number of threads. Try to use all available threads (i.e., numCores * SMT)." +ENDC
        ttcArgs.numThreads = multiprocessing.cpu_count()
        print WARNING + "   => Using %d threads now"%ttcArgs.numThreads + ENDC


    #prefetchDistance
    if(len (ttcArgs.prefetchDistances) == 0):
        ttcArgs.prefetchDistances.append(5) #TODO get best prefetch distance from DB

    #remove duplicates
    ttcArgs.prefetchDistances = set(ttcArgs.prefetchDistances)

    #loopPermutation
    if( len(ttcArgs.loopPermutations) == 0):
        start = 0
        if( ttcArgs.idxPerm[0] == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
            start = 1
        for loopPerm in itertools.permutations(range(start,len(ttcArgs.idxPerm))):
            loopPerm = list(loopPerm)
            ttcArgs.loopPermutations.append(loopPerm)
    else:
        if( ttcArgs.idxPerm[0] == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
            for loopPerm in ttcArgs.loopPermutations:
                loopPerm.remove(0)



    ###########################################
    # sanity check
    ###########################################

    if( ttcArgs.align == 0 and ttcArgs.architecture == "knc"):
            print FAIL + "[TTC] ERROR: non-aligned transpositions are not supported for KNC" + ENDC
            exit(-1)
    if( len(ttcArgs.ldb) != 0 and len(ttcArgs.ldb) != len(ttcArgs.size)):
            print FAIL + "[TTC] ERROR: not all leading dimensions of B have been specified" + ENDC
            exit(-1)
    if( len(ttcArgs.lda) != 0 and len(ttcArgs.lda) != len(ttcArgs.size)):
            print FAIL + "[TTC] ERROR: not all leading dimensions of A have been specified" + ENDC
            exit(-1)
    for i in range(len(ttcArgs.ldb)):
        if(ttcArgs.size[ttcArgs.idxPerm[i]] > ttcArgs.ldb[i]):
            print FAIL + "[TTC] ERROR: the leading dimension of B for dim %d is smaller than allowed (it must be >= %d)"%(i,ttcArgs.size[ttcArgs.idxPerm[i]]) + ENDC
            exit(-1)
    for i in range(len(ttcArgs.lda)):
        if(ttcArgs.size[i] > ttcArgs.lda[i]):
            print FAIL + "[TTC] ERROR: the leading dimension of A for dim %d is smaller than allowed (it must be >= %d)"%(i,ttcArgs.size[i]) + ENDC
            exit(-1)
    
    for prefetchDistance in ttcArgs.prefetchDistances:
        if( prefetchDistance < 0 ):
            print FAIL + "[TTC] ERROR: prefetch distance needs to be positive." + ENDC
            exit(-1)

    sortedPerm = sorted(ttcArgs.idxPerm)
    for i in range(_dim):
        if sortedPerm[i] != i:
            print FAIL + "[TTC] ERROR: permutation is invalid." + ENDC
            exit(-1)

    for loopPerm in ttcArgs.loopPermutations:
        sortedLoopPerm = sorted(loopPerm)
        for i in range(len(loopPerm)):
            start = 0
            if( ttcArgs.idxPerm[0] == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
                start = 1
            if sortedLoopPerm[i] != i + start:
                print FAIL + "[TTC] ERROR: Loop permutation is invalid." + ENDC
                print loopPerm, sortedLoopPerm,i
                exit(-1)

    transposeName = getTransposeName(ttcArgs)

    ###########################################
    # Print settings
    ###########################################
    if( ttcArgs.silent != 1):
        print "--------------Settings---------------------"
        print "#threads: ".ljust(20)+"%d"%ttcArgs.numThreads
        if(ttcArgs.compiler == "g++"):
            print "thread affinity: ".ljust(20)+"GOMP_CPU_AFFINITY=%s"%ttcArgs.affinity
        else:
            print "thread affinity: ".ljust(20)+"KMP_AFFINITY=%s"%ttcArgs.affinity
        print "Compiler: ".ljust(20) + ttc_util.getCompilerVersion(ttcArgs.compiler)
        print "-------------------------------------------"

    ###########################################
    # check if a solution already exists
    ###########################################
    connection = sqlite3.connect(_database)
    cursor = connection.cursor()

    #create tables, if necessary
    createTables(cursor)

    fastestVersionBW = 0
    solutionFound = 0
    sizeId = sql_util.getSizeId(cursor, ttcArgs.size)
    if(ttcArgs.ignoreDatabase == 0 and sizeId != -1): #TODO: also lookup streamingstore optimization
        permId = sql_util.getPermId(cursor, ttcArgs.idxPerm)
        if(permId != -1):
            measurementId = sql_util.getMeasurementId(cursor, sizeId, permId,
                    ttcArgs.floatTypeA, ttcArgs.floatTypeB, ttcArgs.beta,
                    ttcArgs.numThreads, ttcArgs.hotA, ttcArgs.hotB,
                    ttc_util.getArchitecture(ttcArgs.architecture)) 
            if( measurementId != -1 ):
                ret = sql_util.getBestImplementation(cursor, measurementId)
                if(ret != -1):
                    blockA = ret[0]
                    blockB = ret[1]
                    loopId = ret[2]
                    prefetchDistance = ret[3]
                    fastestVersionBW = ret[4]
                    if(fastestVersionBW <= 0):
                        print "ERROR: bandwidth not stored in database\n"
                        traceback.print_stack()   
                        exit(-1)
                    ret = sql_util.getLoopPermFrom(cursor, loopId, len(ttcArgs.size))
                    if( ttcArgs.idxPerm[0] == 0 ): #the first index will always be within our kernel (i.e., it will always be the inner-most loop)
                        ret.remove(0)
                    elif( blockA == 1 and blockB == 1):
                        blockA = 8 #quick bugfix (only happens if the reference version was the fastest implementation)
                        blockB = 8 #quick bugfix
                    if( ret != -1 ): #we have found a solution
                        ttcArgs.blockings = [(blockA, blockB)]
                        solutionFound = 1
                        ttcArgs.loopPermutations = [ret]
                        ttcArgs.prefetchDistances = [ prefetchDistance ]
                        if( ttcArgs.silent != 1):
                            print "Solution already exists: generating the solution which was found previously."
                            printEpilog(transposeName, ttcArgs)

#    if( solutionFound == 0):
#        print sizeId, ttcArgs.size, ttcArgs.idxPerm
#        exit(-1)
    ###########################################
    # generate all versions
    ###########################################
    if( ttcArgs.silent != 1):
        print "[generate] Generate all versions"
    t0 = _time.time()

    _parallelize = ttcArgs.numThreads > 1

    if(ttcArgs.compiler == "nvcc"):
        generator = gputg.GPUtransposeGenerator(ttcArgs.idxPerm, ttcArgs.loopPermutations, ttcArgs.size, ttcArgs.alpha, ttcArgs.beta,
                ttcArgs.maxNumImplementations, ttcArgs.floatTypeA, ttcArgs.floatTypeB, ttcArgs.blockings, _noTest ,ttcArgs.vecLength , ttcArgs.lda, ttcArgs.ldb)
    else:		
        generator = tg.transposeGenerator(ttcArgs.idxPerm, ttcArgs.loopPermutations, ttcArgs.size, ttcArgs.alpha, ttcArgs.beta,
                ttcArgs.maxNumImplementations, ttcArgs.floatTypeA, ttcArgs.floatTypeB, _parallelize, ttcArgs.streamingStores,
                ttcArgs.prefetchDistances, ttcArgs.blockings, _papi, _noTest, ttcArgs.scalar, ttcArgs.align,
                ttcArgs.architecture, _mpi, ttcArgs.lda, ttcArgs.ldb, ttcArgs.silent, ttcArgs.hotA, ttcArgs.hotB )


    generator.generate()
    numSolutions = generator.getNumSolutions() + 1#account for reference version
    if( ttcArgs.silent != 1):
        print "Generation of %d implementations took %f seconds "%(numSolutions,_time.time() - t0)

    emitReference = 0
    measuringTime = 0
    compilationTime = 0
    if _generateOnly == 0:
        if( solutionFound == 0 ): #only compile and measure if we have not found the best solution in our database yet
            if( numSolutions > 1):
                ###########################################
                # compile all versions
                ###########################################
                if( ttcArgs.silent != 1):
                    print "[make] Compile all versions"
                t0 = _time.time()
                numThreadsCompile = max(2, multiprocessing.cpu_count()/2)
                if ttcArgs.debug == 0:
                    if( _mpi ):
                        if( ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512" ):
                            print "[TTC] ERROR: knc + mpi not supported yet."
                            exit(-1)
                        ret = subprocess.call(["make", "-j%d"%numThreadsCompile , "mpi"], stdout=DEVNULL, stderr=subprocess.STDOUT)
                    else:
                        if( ttcArgs.architecture == "knc" ):
                            ret = subprocess.call(["make", "-j%d"%numThreadsCompile, ttcArgs.architecture], stdout=DEVNULL, stderr=subprocess.STDOUT)
                        else:
                            ret = subprocess.call(["make", "-j%d"%numThreadsCompile, ttcArgs.compiler], stdout=DEVNULL, stderr=subprocess.STDOUT)
                else:
                    if( _mpi ):
                        if( ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512" ):
                            print "[TTC] ERROR: knc + mpi not supported yet."
                            exit(-1)
                        ret = subprocess.call(["make", "-j%d"%numThreadsCompile, "mpi"])
                    else:
                        if( ttcArgs.architecture == "knc"  or ttcArgs.architecture == "avx512"):
                            ret = subprocess.call(["make", "-j%d"%numThreadsCompile, ttcArgs.architecture])
                        else:
                            ret = subprocess.call(["make", "-j%d"%numThreadsCompile, ttcArgs.compiler])
                if ret != 0 :
                    print FAIL+"[TTC] [Error] compilation failed. Retry with '-v' option to see the compilation errors." + ENDC
                    exit(-1)
                compilationTime = (_time.time() - t0)
                if( ttcArgs.silent != 1):
                    print "Compilation took %f seconds"%compilationTime

                ###########################################
                # run versions
                ###########################################
                if( ttcArgs.silent != 1):
                    print "[running] measure runtime"

                #set environment variables
                my_env = os.environ.copy()
                if(ttcArgs.compiler != "nvcc"):
                    my_env["OMP_NUM_THREADS"] = str(ttcArgs.numThreads)
                    if(ttcArgs.compiler == "g++"):
                        my_env["GOMP_CPU_AFFINITY"] = ttcArgs.affinity
                    else:
                        my_env["KMP_AFFINITY"] = ttcArgs.affinity 

                t0 = _time.time()
                outputTiming = []
                if( _mpi ):
                    _numSockets = 2
                    proc = subprocess.Popen(["mpirun", "-n","%d"%_numSockets, "-env", "I_MPI_PIN", "1", "-env", "KMP_AFFINITY=verbose,compact", "-env", "OMP_NUM_THREADS=%d"%(ttcArgs.numThreads/_numSockets), "-env","I_MPI_PIN_DOMAIN=socket", "-env","I_MPI_PIN_CELL=core","./transpose.exe"],stderr=subprocess.STDOUT,stdout=subprocess.PIPE, env=my_env)
                else:
                    if( ttcArgs.architecture == "knc"):
                        proc = subprocess.Popen(["ssh_mic","source /etc/profile; KMP_AFFINITY=%s OMP_NUM_THREADS=%d  %s/transpose.exe"%(ttcArgs.affinity,ttcArgs.numThreads,_ttc_root)],stderr=subprocess.STDOUT,stdout=subprocess.PIPE, env=my_env, stdin=subprocess.PIPE)
                    else:
                        proc = subprocess.Popen(['./transpose.exe'],stderr=subprocess.STDOUT,stdout=subprocess.PIPE, env=my_env)
                counter = 0
                counter2 = 0
                failCount = 0
                while True:
                    line = proc.stdout.readline()
                    line = line.lower()
                    outputTiming.append(line)

                    counter2 +=1 
                    if(line.find("variant") != -1 and line.find("took") != -1):
                        counter +=1 
                    if( line.find("error") != -1 ):
                        print FAIL + line + ENDC
                        _logFile.write(FAIL + line + ENDC)
                        failCount += 1
                        break

                    if( counter2 > (numSolutions*2+10) or line.find("top-5") != -1 ):
                        break

                    if ttcArgs.debug:
                        print "%d / %d :"%(counter,numSolutions) + line[:-1]
                        sys.stdout.flush()
                    else:
                        if( ttcArgs.silent != 1):
                            sys.stdout.write("[TTC] %d out of %d implementations done.\r"%(counter,numSolutions))
                        sys.stdout.flush()

                proc.wait()
                if proc.returncode != 0:
                    print proc.poll()
                    print FAIL+"[Error] runtime error.", ENDC
                    exit(-1)

                if failCount == 0:
                    if( ttcArgs.silent != 1):
                        print OKGREEN + "[Success] all %d tests passed."%numSolutions, ENDC
                else:
                    print FAIL + "[TTC_Error] %d out of %d tests failed."%(failCount, numSolutions), ENDC

                fastestVersion = "-1"
                fastestVersionTime = 1000000000000000.0
                fastestVersionBW = 0
                top1Speedup = 0
                top5Speedup = 0
                referenceBw = -1
                #pick fastest version
                for line in outputTiming:
                    tokens = line.split()
                    if( line.find("reference version") != -1 ):
                        referenceBw = float(tokens[7])
                        if( float(tokens[4]) < fastestVersionTime ):
                            fastestVersion = tokens[2]
                            fastestVersionBW = float(tokens[7])
                            #if( ttcArgs.silent != 1):
                            #    print "Reference version attains %f GiB/s."%(referenceBw)
                    elif( len(tokens) >= 7  and tokens[0] == "variant" and tokens[2] == "took" ):
                        #if( ttcArgs.silent != 1):
                        #    print "Version %s attains %f GiB/s."%(tokens[1],float(tokens[6]))
                        if( float(tokens[3]) < fastestVersionTime ):
                            fastestVersion = tokens[1]
                            fastestVersionTime = float(tokens[3])
                            fastestVersionBW = float(tokens[6])
                    if( len(tokens) == 3 and tokens[0] == "Top-1" ):
                        top1Speedup = float(tokens[2])
                    if( len(tokens) == 3 and tokens[0] == "Top-5" ):
                        top5Speedup = float(tokens[2])

                if(fastestVersionBW < referenceBw and ttcArgs.compiler!="nvcc" ): #fallback to reference version if this is the fastest
                    fastestVersionBW = referenceBw
                    fastestVersion = "reference"
                    emitReference = 1 #force generation of reference version (see line 945)

                    #build string for reference version
                    variant = "v"
                    for i in ttcArgs.idxPerm[-1::-1]:
                        variant += str(i)
                    variant += "_1x1"
                    outputTiming.append("variant %s took -1 and achieved %.2f GiB/s (blocking rank: 0) (loop rank: 0) (l2 misses: 0) (invalidates: 0)"%(variant,fastestVersionBW))

                measuringTime = (_time.time() - t0)
                if( ttcArgs.silent != 1):
                    print "Measuring took %f seconds"%measuringTime
                    if( referenceBw > 0 ):
                        print "Speedup over reference: %.2f"%(fastestVersionBW/referenceBw)
                    print "\nThe fastest version (%s) attains %.2f GiB/s.\n"%(fastestVersion,fastestVersionBW)

        ###########################################
        # save fastest version to file
        ###########################################
        if( solutionFound == 1):
            fastestVersion = generator.implementations[-1].getVersionName()
        if( emitReference or numSolutions == 1):
            code = generator.referenceImplementation.getImplementation(_parallelize, 1)
            transposeName = generator.referenceImplementation.getTransposeName(1)
        else: 
            code = generator.generateVersion(fastestVersion)
        if( len(code) > 1):
            directory = workingDir +"/ttc_transpositions"
            if not os.path.exists(directory):
                os.makedirs(directory)
            if ttcArgs.architecture == "avx" or ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512":
                generator.generateOffsetFile(directory)
	    elif(ttcArgs.architecture == "cuda"):
		cppFile = directory + "/%s.cu"%transposeName
                f = open(cppFile ,'w')
                cppCode = code[0]
                f.write(cppCode)
                f.close()
            hFile = directory + "/%s.h"%transposeName

            define = transposeName + "_H"
            hppCode = "#ifndef %s\n"%(define.upper())
            hppCode += "#define %s\n"%(define.upper())
            if( ttcArgs.scalar != 1 ):
                if ttcArgs.architecture == "avx" or ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512":
                    hppCode += "#include <xmmintrin.h>\n#include <immintrin.h>\n"
                elif ttcArgs.architecture == "power":
                    hppCode += "#include <builtins.h>\n"
                    hppCode += "#include <altivec.h>\n"
            if ttcArgs.floatTypeA.find("complex") != 1 or ttcArgs.floatTypeB.find("complex") != 1:
                hppCode += "#include <complex.h>\n"
            f = open(hFile,'w')
            if ttcArgs.architecture == "avx" or ttcArgs.architecture == "knc" or ttcArgs.architecture == "avx512":
                hppCode += code
	    elif(ttcArgs.architecture == "cuda"):
                hppCode += code[1]
            hppCode += "#endif\n"
            f.write(hppCode)
            f.close()

            if( ttcArgs.silent != 1):
                printEpilog(transposeName, ttcArgs)


        ###########################################
        # update Database
        ###########################################
        if( ttcArgs.updateDatabase and solutionFound == 0 and numSolutions > 1):

            version = getVersion()
            host = socket.gethostname()
            dim = len(ttcArgs.idxPerm)

            #######################
            # insert all measurements into the database ##############
            #######################

            #update size table
            size_id = insertIntoSize(cursor, ttcArgs.size, _logFile )

            #update perm table
            perm_id = insertIntoPermutation(cursor, ttcArgs.idxPerm, _logFile )

            #update measurements table
            measurement_id = insertIntoMeasurements(cursor,dim, host, version, ttcArgs.alpha,
                    ttcArgs.beta, ttcArgs.numThreads, compiler_version, ttcArgs.floatTypeA, ttcArgs.floatTypeB,0,
                    0, compilationTime, measuringTime, size_id, perm_id,
                    _logFile, referenceBw, ttcArgs.hotA, ttcArgs.hotB, ttc_util.getArchitecture(ttcArgs.architecture)) #ttcArgs.architecture)


            for line in outputTiming:
                tokens = line.split()
                #update record
                if( len(tokens) >= 18  and tokens[0] == "variant" and tokens[2] == "took" ):
                    (blockA, blockB, prefetchDistance, loopOrder) = getInfoAboutVersion(tokens[1])
                    time = float(tokens[3])
                    BW = float(tokens[6])
                    rankBlocking = int(tokens[10][:-1])#remove last ')'
                    rankLoopOrder =  int(tokens[13][:-1])#remove last ')'
                    tlbMisses = -1.0 #remove last ')'
                    l2misses = float(tokens[16][:-1])#remove last ')'

                    #update perm table
                    loop_id = insertIntoLoopOrder (cursor, loopOrder, _logFile )

                    #update variant table
                    insertIntoVariant(cursor,
                            blockA,blockB,prefetchDistance,BW,time, rankLoopOrder,
                            rankBlocking, measurement_id, loop_id, tlbMisses, l2misses, _logFile )


            connection.commit()
            connection.close()

    os.chdir(workingDir)
    _logFile.close()
    return (transposeName,fastestVersionBW )

def main():

    _allowedArguments = [ "--compiler","--use-streamingStores","--maxImplementations",
            "--help","--alpha","--beta","--papi","--size","--perm", "--loopPerm","--dataType",
            "--numThreads", "--generateOnly","--prefetchDistances",
            "--updateDatabase","--dontCompile","-v", "--blockings",
            "--noTest","--no-align","--no-vec","--mpi", "--architecture",
            "--affinity","--lda", "--ldb", "--ignoreDatabase", "--threadsPerBlock", "--hotA", "--hotB"]

    _hotA = 0
    _hotB = 0
    _ignoreDatabase = 0
    _affinity = ""
    _scalar = 0
    _updateDatabase = 1
    _database = "ttc.db"
    _prefetchDistances = []
    _papi = 0
    _numThreads = 0
    _streamingStores = 0
    _idxPerm = []
    _size = []
    _lda = []
    _ldb = []
    _alpha = 1.
    _beta = 0.
    _maxNumImplementations = 200
    _showHelp = 0
    _floatTypeA = "float"
    _floatTypeB = "float"
    _debug = 0
    _mpi = 0
    _compiler = "icpc"
    _architecture= "avx"
    _generateOnly = 0
    _logFile = open("log.txt","a+")
    _blockings = []
    _noTest = 0
    _loopPermutations = []
    _align = 1
    _vecLength = []


    ###########################################
    # parse arguments
    ###########################################
    for arg in sys.argv:
        if( arg == sys.argv[0]): continue

        valid = 0
        for allowed in _allowedArguments:
            if arg.split("=")[0] == allowed:
                valid = 1
                break;
        if(valid == 0):
            printHelp()
            print FAIL + "Error: argument "+arg.split("=")[0] + " not valid." + ENDC
            exit(-1)

        if arg == "--ignoreDatabase":
            _ignoreDatabase = 1
        if arg == "--mpi":
            _mpi = 1
        if arg == "--no-align":
            _align = 0
        if arg == "--no-vec":
            _scalar = 1
        if arg == "--noTest":
            _noTest = 1
        if arg == "--updateDatabase":
            _updateDatabase = 1
        if arg == "--use-streamingStores":
            _streamingStores = 1
        if arg == "-v":
            _debug = 1
        if arg == "--verbose":
            _debug = 1
        if arg == "--generateOnly":
            _generateOnly = 1
        if arg == "--hotA":
            _hotA = 1
        if arg == "--hotB":
            _hotB = 1
        if arg == "--papi":
            _papi = 1
        if arg.find("--architecture=") != -1:
            if( arg.split("=")[1] == "avx" ):
                _architecture = "avx"
            elif( arg.split("=")[1] == "knc" ):
                _architecture = "knc"
            elif( arg.split("=")[1] == "avx512" ):
                _architecture = "avx512"
            elif( arg.split("=")[1] == "power" ):
                _architecture = "power"
            elif( arg.split("=")[1] == "cuda" ):
                _architecture = "cuda"
                _compiler="nvcc"
            else:
                print ERROR + "[TTC] ERROR: Architecture unknown." +ENDC
                exit(-1)
        if arg == "--help":
            _showHelp = 1
        if arg.find("--compiler=") != -1:
            if( arg.split("=")[1] == "icpc" ):
                _compiler = "icpc"
            elif( arg.split("=")[1] == "g++" ):
                _compiler = "g++"
            elif( arg.split("=")[1] == "ibm" ):
                _compiler = "ibm"
                _architecture = "power"
            elif( arg.split("=")[1] == "nvcc" ):
                _compiler = "nvcc"
                _architecture = "cuda"
            else:
                print FAIL + "[TTC] ERROR: unknown compiler choice." + ENDC
                exit(-1)
        if arg.find("--dataType=") != -1:
            if( arg.split("=")[1] == "s" ):
                _floatTypeA = "float"
                _floatTypeB = "float"
            elif( arg.split("=")[1] == "d" ):
                _floatTypeA = "double"
                _floatTypeB = "double"
            elif( arg.split("=")[1] == "c" ):
                _floatTypeA = "float complex"
                _floatTypeB = "float complex"
            elif( arg.split("=")[1] == "z" ):
                _floatTypeA = "double complex"
                _floatTypeB = "double complex"
            elif( arg.split("=")[1] == "sd" ):
                _floatTypeA = "float"
                _floatTypeB = "double"
            elif( arg.split("=")[1] == "ds" ):
                _floatTypeA = "double"
                _floatTypeB = "float"
            elif( arg.split("=")[1] == "cz" ):
                _floatTypeA = "float complex"
                _floatTypeB = "double complex"
            elif( arg.split("=")[1] == "zc" ):
                _floatTypeA = "double complex"
                _floatTypeB = "float complex"
            else:
                print "[TTC] ERROR: unknown precision. It needs to be either of the following s, d, c, z, sd, ds, cz, or zc."
                exit(-1)
        if arg.find("--affinity=") != -1:
            _affinity = arg.split("=")[1]
        if arg.find("--numThreads=") != -1:
            _numThreads = int(arg.split("=")[1]) 
        if arg.find("--maxImplementations=") != -1:
            maxNumVersions = int(arg.split("=")[1]) 
            _maxNumImplementations = maxNumVersions
        if arg.find("--alpha=") != -1:
            _alpha = float(arg.split("=")[1])
        if arg.find("--beta=") != -1:
            _beta = float(arg.split("=")[1])
        if arg.find("--prefetchDistances=") != -1:
            prefetchDistances = arg.split("=")[1].split(",")
            for prefetchDistance in prefetchDistances:
                _prefetchDistances.append(int(prefetchDistance))
        if arg.find("--blockings=") != -1:
            blockings = arg.split("=")[1].split(",")
            for blocking in blockings:
                _blockings.append((int(blocking.split("x")[0]),int(blocking.split("x")[1])))

        if arg.find("--threadsPerBlock=") != -1:
            vectorLengths = arg.split("=")[1].split(",")
            for v in vectorLengths:
                if(int(v) != 256 and int(v) != 128 and int(v) != 512):
                    print "[TTC] ERROR: value for --threadsPerBlock is not valid. It needs to be either 128, 256 or 512."
                    exit(-1)
                else:
                    _vecLength.append(int(v))

        if arg.find("--size=") != -1:
            sizes = arg.split("=")[1]
            _size = []
            for s in sizes.split(","):
                _size.append(int(s))
            _dim = len(_size)
        if arg.find("--lda=") != -1:
            lda = arg.split("=")[1]
            for size in lda.split(","):
                _lda.append(int(size))
        if arg.find("--ldb=") != -1:
            ldb = arg.split("=")[1]
            for size in ldb.split(","):
                _ldb.append(int(size))
        if arg.find("--perm=") != -1:
            perm = arg.split("=")[1]
            _idxPerm = []
            for idx in perm.split(","):
                _idxPerm.append(int(idx))
            _dim = len(_idxPerm)

        if arg.find("--loopPerm=") != -1:
            tmp = arg.split("=")[1]
            for loopPermStr in tmp.split("-"):
                loopPerm = []
                for loopIdx in loopPermStr.split(","):
                    loopPerm.append(int(loopIdx))
                _loopPermutations.append(loopPerm)


    if( len(_idxPerm) != len(_size) ):
        printHelp()
        print FAIL+"[Error] dimension of permutation and size does not match." + ENDC
        exit(-1)

    if( len(sys.argv) < 2 or _showHelp ):
        printHelp()
        exit(0)

    ttc_args = ttc_util.TTCargs(_idxPerm, _size)
    ttc_args.alpha = _alpha
    ttc_args.beta = _beta
    ttc_args.affinity = _affinity
    ttc_args.numThreads = _numThreads
    ttc_args.floatTypeA = _floatTypeA
    ttc_args.floatTypeB = _floatTypeB
    ttc_args.streamingStores = _streamingStores
    ttc_args.maxNumImplementations = _maxNumImplementations
    ttc_args.lda = _lda
    ttc_args.ldb = _ldb
    ttc_args.debug = _debug
    ttc_args.architecture = _architecture
    ttc_args.align = _align
    ttc_args.blockings = _blockings
    ttc_args.loopPermutations = _loopPermutations
    ttc_args.prefetchDistances = _prefetchDistances
    ttc_args.scalar = _scalar
    ttc_args.compiler  = _compiler 
    ttc_args.vecLength = _vecLength
    ttc_args.silent = 0
    ttc_args.hotA = _hotA
    ttc_args.hotB = _hotB
    ttc_args.ignoreDatabase = _ignoreDatabase

    generateTransposition( ttc_args )


