import sqlite3

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

def getSizeId(cursor, size):
    dim = len(size)
    sizeStr = ""
    for i in range(dim):
        sizeStr += "size%d = %d"%(i,size[i])
        sizeStr += " and "
    sizeStr += "size%d is NULL"%dim

    command = """
              select size_id from size where %s;
              """%(sizeStr)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        return -1

    result = cursor.fetchall() 
    if( len(result) > 0):
        return result[0][0]
    return -1


def getPermId(cursor, perm):
    dim = len(perm)
    permStr = ""
    for i in range(dim):
        permStr += "idx%d = %d"%(i,perm[i])
        permStr += " and "
    permStr += "idx%d is NULL"%dim

    command = """
              select permutation_id from permutation where %s;
              """%(permStr)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        return -1

    result = cursor.fetchall() 
    if( len(result) > 0):
        return result[0][0]
    return -1

def getMeasurementId(cursor, sizeId, permId, floatTypeA, floatTypeB, beta, numThreads, hotA, hotB, architecture):
    betaStr = ""
    if( beta != 0):
        betaStr = "beta != 0"
    else:
        betaStr = "beta = 0"
    command = """
              select measurement_id from measurements where size_id = %d and
              permutation_id = %d and floatTypeA = '%s' and floatTypeB = '%s' and %s
              and numThreads >= %d and hotA = %d and hotB = %d and architecture='%s';
              """%(sizeId, permId, floatTypeA, floatTypeB, betaStr, numThreads, hotA, hotB, architecture) 
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        return -1

    result = cursor.fetchall() 
    if( len(result) > 0):
        return result[0][0]

    return -1

def getBestImplementation(cursor, measurementId):
    command = """
              select blockA, blockB, loopOrder_id, prefetchDistance, bandwidth from variant 
              where measurement_id = %d order by bandwidth DESC;
              """%(measurementId)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print command
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        return -1

    result = cursor.fetchall() 
    if( len(result) > 0):
        return (result[0][0],result[0][1],result[0][2],result[0][3],result[0][4])

    return -1


def getLoopPermFrom(cursor, loopId, dim):
    indices = ""
    for i in range(dim):
        indices += "idx%d"%(i)
        if( i != dim -1 ):
            indices += ", "

    command = """
              select %s from loopOrder where loopOrder_id = %d;
              """%(indices, loopId)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print commmand
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        return -1

    result = cursor.fetchall() 
    if( len(result) > 0):
        perm = []
        for i in range(dim):
            perm.append(result[0][i])
        return perm 
    return -1
























