
import sys
import os
import sqlite3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math


OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'


database = "ttc.db"
_host = "linuxihdc077_knc"

_constraint = """
          floatType = 'float' 
          """

#Example:
#select MAX(bandwidth), prefetchDistance  from variant where measurement_id = 11 GROUP BY prefetchDistance;

def listToString(perm):
    string = ""
    for s in perm:
        string += str(s) + ","
    return string[:-1]



#slowdown due to sw prefetching for the FASTEST versions
def getBestPrefetchDistance(cursor):

    #get all variants and their prefetchSpeedups for the given constraints:
    command = """
        select measurement_id, prefetchDistance, max(bandwidth) from variant where measurement_id in (select measurement_id from measurements where %s) group by measurement_id, prefetchDistance
    """%_constraint
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    result = cursor.fetchall() 
    measurements = {}
    distances = {}
    for r in result:
        measurement_id = r[0]
        prefetchDistance = r[1]
        bandwidth = r[2]
        if( not distances.has_key(prefetchDistance) ):
            distances[prefetchDistance] = prefetchDistance
        if( measurements.has_key(measurement_id) ):
            measurements[measurement_id][prefetchDistance] = bandwidth
        else:
            measurements[measurement_id] = {}
            measurements[measurement_id][prefetchDistance] = bandwidth

    tmp = []
    for d in distances:
        tmp.append(d)
    tmp = list(set(tmp))
    tmp.sort()
    distances = tmp
    maxDistance = max(distances)
    minDistance = min(distances)

    counts = {}
    slowdowns = {}
    for measurement in measurements:
        maxBw = -1
        bestDistance = -1
        if(len(measurements[measurement]) > 1): #exclude those permutations for which no sw prefetching is supported
            for prefetchDistance in measurements[measurement]:
                if( maxBw < measurements[measurement][prefetchDistance] ):
                    maxBw = measurements[measurement][prefetchDistance]
                    bestDistance = prefetchDistance

            for prefetchDistance in measurements[measurement]:
                if( slowdowns.has_key(prefetchDistance) ):
                    slowdowns[prefetchDistance].append( measurements[measurement][prefetchDistance] / maxBw)
                else:
                    slowdowns[prefetchDistance] = [ measurements[measurement][prefetchDistance] / maxBw ]

            if( counts.has_key(bestDistance) ):
                counts[bestDistance] += 1
            else:
                counts[bestDistance] = 1

    tmpCounts = [0 for d in range(maxDistance - minDistance+1)]
    for bestDistance in counts:
        tmpCounts[bestDistance - minDistance] = counts[bestDistance]
    totalCount = sum(tmpCounts)
    for i in range(len(tmpCounts)):
        tmpCounts[i] = float(tmpCounts[i])/totalCount * 100

    minSpeedups = []
    maxSpeedups = []
    avgSpeedups = []
    stdSpeedups = []
    for i in range(len(tmpCounts)):
        prefetchDistance = i + minDistance
        minSlowdown = min(slowdowns[prefetchDistance])
        maxSlowdown = max(slowdowns[prefetchDistance])
        arr = np.array(slowdowns[prefetchDistance])
        avgSlowdown = np.mean(arr)
        stdSlowdown = np.std(arr)
        minSpeedups.append(minSlowdown)
        maxSpeedups.append(maxSlowdown)
        avgSpeedups.append(avgSlowdown)
        stdSpeedups.append(stdSlowdown)
        #print prefetchDistance, tmpCounts[i], minSlowdown, maxSlowdown, avgSlowdown


    #create plot
    fig, ax1 = plt.subplots()
    plt.grid(axis='y')
    ax1.set_axis_bgcolor((248/256., 248/256., 248/256.))
    ax1.plot(distances, minSpeedups, label= "min", marker = 'v' , linewidth=2.0, markeredgewidth = 2.0, clip_on=False)
    ax1.errorbar(distances, avgSpeedups,stdSpeedups, label= "avg", marker = 'x' , linewidth=2.0, markeredgewidth = 2.0, clip_on=False)
    ax1.plot(distances, maxSpeedups, label= "max", marker = '^' , linewidth=2.0, markeredgewidth = 2.0, clip_on=False)
    ax1.set_ylim(top=1)
    ax2 = ax1.twinx()
    ax2.plot(distances, tmpCounts, label="count", linestyle='--', color='#000000', marker = 'x' , linewidth=2.0, markeredgewidth = 2.0, clip_on=False)
    ax2.set_ylabel('Relative count [%]')
    ax1.set_xlabel('Prefetch distance')
    ax1.set_ylabel('Speedup')
    ax1.legend(loc='lower right')
    ax2.legend(loc='center right')
    outputDir = "./plots"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    plt.savefig(outputDir + "/bestPrefetchDistance.pdf", bbox_inches='tight')
    plt.close()


def createBenchmarkPerfPlots(cursor):

    constraint = _constraint

    command = """
    select dim, bandwidth, ifnull(referenceBandwidth,0) from fastest0Percent join size on fastest0Percent.size_id = size.size_id where
            %s;
    """ % constraint
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    result = cursor.fetchall() 
    bandwidth = {}
    bandwidth_nodim = []
    speedup_nodim = []
    speedup = {} #over compiler
    for r in result:
        bandwidth_nodim.append(r[1])
        if(r[2] > 0):
            speedup_nodim.append(r[1]/r[2])
        dim = r[0]
        if( bandwidth.has_key(dim) ):
            bandwidth[dim].append(r[1])
            if(r[2] > 0):
                speedup[dim].append(r[1]/r[2])
        else:
            bandwidth[dim] = [r[1]]
            if(r[2] > 0):
                speedup[dim] = [r[1]/r[2]]

    maxList = []
    minList = []
    avgList = []
    stdList = []
    maxSpeedup = []
    minSpeedup = []
    avgSpeedup = []
    stdSpeedup = []
    xList = []
    for dim in bandwidth:
        arr = np.array(bandwidth[dim])
        std = np.std(arr)
        maxBW = max(bandwidth[dim])
        minBW = min(bandwidth[dim])
        avgBW = float(sum(bandwidth[dim])) / len(bandwidth[dim])

        xList.append(dim)
        minList.append(minBW)
        avgList.append(avgBW)
        maxList.append(maxBW)
        stdList.append(std)

        arrSpeedup = np.array(speedup[dim])
        stdSP = np.std(arrSpeedup)
        maxSP = max(speedup[dim])
        minSP = min(speedup[dim])
        avgSP = float(sum(speedup[dim])) / len(speedup[dim])
        #print "%d, %.2f, %.2f, %.2f, %.2f"%(dim, maxBW, minBW, avgBW, std)
        maxSpeedup.append( maxSP )
        minSpeedup.append( minSP )
        avgSpeedup.append( avgSP )
        stdSpeedup.append( stdSP )

################################# BAR PLOTS #############################
    x = np.array(xList)
    offset = 0.2
    x = x +offset
    width = 0.15       # the width of the bars
    fig, ax = plt.subplots()
    plt.grid(axis='y', zorder=0)
    ax.set_axis_bgcolor((248/256., 248/256., 248/256.))
    rects1 = ax.bar(x+width, maxList, width, color='#2c7fb8', zorder=3)
    rects2 = ax.bar(x + 2*width, avgList, width, color='#7fcdbb', zorder=3)
    rects3 = ax.bar(x + 3*width, minList, width, color='#edf8b1', zorder=3)
    ax.set_ylabel('Bandwidth [GiB/s]')
    ax.set_xlabel('Dimension')
    ax.set_xticks(x+ 2*width)
    #ax.set_yticks(np.arange(0, 161, 20))

    ax.set_xticklabels(('2','3','4','5'))
    ax.legend((rects1[0], rects2[0],rects3[0]), ('Max', 'Avg', 'Min'),
            loc ='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow = True, ncol = 4)
    plt.savefig(outputDir + "/benchmark_bandwidth.pdf", bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots()
    plt.grid(axis='y', zorder=0)
    ax.set_axis_bgcolor((248/256., 248/256., 248/256.))
    rects1 = ax.bar(x, maxSpeedup, width, color='#2c7fb8', zorder=3)
    rects2 = ax.bar(x + width, avgSpeedup, width, color='#7fcdbb', zorder=3)
    rects3 = ax.bar(x + 2*width, minSpeedup, width, color='#edf8b1', zorder=3)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Dimension')
    ax.set_xticks(x+ 1*width)
    #plt.plot([2, 6], [1, 1], 'k--', lw=1)
    #ax.set_yticks(np.arange(0, 31, 2))
    ax.set_xticklabels(('2','3','4','5'))
    ax.legend(( rects1[0], rects2[0],rects3[0]), ('Max', 'Avg', 'Min'),
            loc ='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow = True, ncol = 4)
    plt.savefig(outputDir + "/benchmark_speedup.pdf",bbox_inches='tight')
    plt.close()


###########################################
# open DB
###########################################
if( not os.path.isfile(database) ):
    print "Error cannot open database %s.\n"%database
    exit(-1) 

_connection = sqlite3.connect(database)
_cursor = _connection.cursor()


###########################################
# parse arguments
###########################################

outputDir = "./plots"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

createBenchmarkPerfPlots(_cursor)
getBestPrefetchDistance(_cursor)
