import geopy.distance
import gpxpy
import gpxpy.gpx
from sklearn.cluster import DBSCAN
from sklearn import metrics
import os
import pandas as pd
import gmplot
import numpy as np
from shapely.geometry import MultiPoint
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import collections

user = '001'
userdata = './Geolife Trajectories 1.3/Data/' + user + '/Trajectory/'

AVERAGE_CONTENT_SIZE = 100
DISTANCE_THRESHOLD = 0.5 # km

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def gen_sentinel_point(start, end, number):
    q = collections.deque()
    q.append((start, end))
    points = set()
    while q and len(points) < number:
        (start, end) = q.pop()
        x1, y1 = start
        x2, y2 = end
        a = min(x1, x2)
        b = min(y1, y2)
        d = max(y1, y2)
        c = max(x1, x2)
        points.add((a, b))
        points.add((c, d))
        points.add((a, d))
        points.add((c, b))
        # plt.figure(figsize=(10,10.1),dpi=125)
        # plt.plot(a,b,'ro')
        # plt.plot(c,d,'ro')
        # plt.plot(a,d,'ro')
        # plt.plot(c,b,'ro')
        np_x = min(x1, x2) + abs(x2 - x1) / 2
        np_y = min(y1, y2) + abs(y2 - y1) / 2
        np = (np_x, np_y)
        q.appendleft((np, (a,b)))
        q.appendleft((np, (c,d)))
        q.appendleft((np, (a,d)))
        q.appendleft((np, (c,b)))
    return points

    #plt.plot(np_x,np_y,'o')
    #plt.show()

#res = gen_sentinel_point((52.2296756, 21.0122287),(52.406374, 16.9251681),84)
# plt.figure(figsize=(10,10.1),dpi=125)
# for item in res:
#     plt.plot(item[0], item[1],'ro')
# plt.show()
# cal distance between two gps points
def within_range_gps(coords_1, coords_2):
    distance =  geopy.distance.distance(coords_1, coords_2).km
    return distance

def tracking(report, res):
    min_distance = 999999
    for ele in res:
        distance = geopy.distance.distance(report, ele).km
        if distance < min_distance:
            min_distance = distance
            result = ele
    return result, min_distance

def get_user_data():
    filelist = os.listdir(userdata)
    names = ['lat','lng','zero','alt','days','date','time']
    df_list = [pd.read_csv(userdata + f,header=6,names=names,index_col=False) for f in filelist]
    df = pd.concat(df_list, ignore_index=True)

    # delete unused column
    df.drop(['zero', 'days'], axis=1, inplace=True)

    # data is recorded every 1~5 seconds, which is too frequent. Reduce it to every minute
    df_min = df.iloc[::12, :]

    df_min.head(10)
    print "Total GPS points:" + str(df_min.shape[0])
    a = np.max(df_min.lat)
    b = np.max(df_min.lng)
    c = np.min(df_min.lat)
    d = np.min(df_min.lng)
    for i in df_min.lat:
        print i
    res = gen_sentinel_point((a, b),(c, d),9)
    plt.figure(figsize=(10,10.1),dpi=125)
    for item in res:
        plt.plot(item[0], item[1],'ro')
    plt.show()
    coords = df_min.as_matrix(columns=['lat', 'lng'])
    corrds_infer = []
    distance_array = []
    for (x, y) in coords:
        #print "AL",(x,y)
        (new_x, new_y), distance = tracking((x,y), res)
        corrds_infer.append((new_x, new_y))
        distance_array.append(distance)
    print corrds_infer
    print distance_array
    # # # declare the center of the map, and how much we want the map zoomed in
    # gmap = gmplot.GoogleMapPlotter(df_min.lat[0], df_min.lng[0], 11)
    # gmap.plot(df_min.lat, df_min.lng)
    # gmap.draw("user001_map.html")
    #
    # # represent GPS points as (lat, lon)
    # coords = df_min.as_matrix(columns=['lat', 'lng'])
    # #coords = np.matrix(corrds_infer)
    # # earth's radius in km
    # kms_per_radian = 6371.0088
    # # define epsilon as 0.5 kilometers, converted to radians for use by haversine
    # epsilon = 0.5 / kms_per_radian
    #
    # # eps is the max distance that points can be from each other to be considered in a cluster
    # # min_samples is the minimum cluster size (everything else is classified as noise)
    # db = DBSCAN(eps=epsilon, min_samples=100, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    # cluster_labels = db.labels_
    # # get the number of clusters (ignore noisy samples which are given the label -1)
    # num_clusters = len(set(cluster_labels) - set([-1]))
    #
    # print 'Clustered ' + str(len(df_min)) + ' points to ' + str(num_clusters) + ' clusters'
    #
    # # turn the clusters in to a pandas series
    # clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    # #print clusters
    # # get the centroid point for each cluster
    # centermost_points = clusters.map(get_centermost_point)
    # lats, lons = zip(*centermost_points)
    # rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
    # fig, ax = plt.subplots(figsize=[10, 6])
    # rs_scatter = ax.scatter(rep_points['lon'][0], rep_points['lat'][0], c='#99cc99', edgecolor='None', alpha=0.7, s=450)
    # ax.scatter(rep_points['lon'][1], rep_points['lat'][1], c='#99cc99', edgecolor='None', alpha=0.7, s=250)
    # ax.scatter(rep_points['lon'][2], rep_points['lat'][2], c='#99cc99', edgecolor='None', alpha=0.7, s=250)
    # ax.scatter(rep_points['lon'][3], rep_points['lat'][3], c='#99cc99', edgecolor='None', alpha=0.7, s=150)
    # df_scatter = ax.scatter(df_min['lng'], df_min['lat'], c='k', alpha=0.9, s=3)
    # ax.set_title('Top N locations infered from ') #Full GPS trace vs. DBSCAN clusters
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.legend([df_scatter, rs_scatter], ['GPS points', 'Cluster centers'], loc='upper right')
    #
    # labels = ['cluster{0}'.format(i) for i in range(1, num_clusters+1)]
    # for label, x, y in zip(labels, rep_points['lon'], rep_points['lat']):
    #     plt.annotate(
    #         label,
    #         xy = (x, y), xytext = (-25, -30),
    #         textcoords = 'offset points', ha = 'right', va = 'bottom',
    #         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
    #         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    #
    # plt.show()
    # # Get the hours for each cluster
    # M = []
    # def myfunc(row):
    #     t = df_min[(df_min['lat']==row[0]) & (df_min['lng']==row[1])]['time'].iloc[0]
    #     return t[:t.index(':')]
    # for i in range(num_clusters):
    #     hours = np.apply_along_axis(myfunc, 1, clusters[i]).tolist()
    #     M.append(map(int, hours))
    # f, axarr = plt.subplots(4, sharex=True, figsize=(6,10))
    # axarr[0].hist(M[0])
    # axarr[0].text(20, 1600, "cluster 1")
    # axarr[1].hist(M[1])
    # axarr[1].text(20, 50, "cluster 2")
    # axarr[2].hist(M[2])
    # axarr[2].text(20, 40, "cluster 3")
    # axarr[3].hist(M[3])
    # axarr[3].text(20, 50, "cluster 4")
    # axarr[3].set_xlabel("Hours of a day")
    # plt.xticks(np.arange(0, 25, 2.0))
    # f.text(0.04, 0.5, '# of GPS points', va='center', rotation='vertical')
    #
    #
    # fig, ax = plt.subplots(figsize=[10, 6])
    # rs_scatter = ax.scatter(rep_points['lon'][0], rep_points['lat'][0], c='#99cc99', edgecolor='None', alpha=0.7, s=450)
    # ax.scatter(rep_points['lon'][1], rep_points['lat'][1], c='#99cc99', edgecolor='None', alpha=0.7, s=250)
    # ax.scatter(rep_points['lon'][2], rep_points['lat'][2], c='#99cc99', edgecolor='None', alpha=0.7, s=250)
    # ax.scatter(rep_points['lon'][3], rep_points['lat'][3], c='#99cc99', edgecolor='None', alpha=0.7, s=150)
    # df_scatter = ax.scatter(df_min['lng'], df_min['lat'], c='k', alpha=0.9, s=3)
    # ax.set_title('Full GPS trace vs. DBSCAN clusters')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.legend([df_scatter, rs_scatter], ['GPS points', 'Cluster centers'], loc='upper right')
    #
    # labels = ['Work', 'Home', 'Home 2']
    # for label, x, y in zip(labels, rep_points['lon'][:num_clusters-1], rep_points['lat'][:num_clusters-1]):
    #     plt.annotate(
    #         label,
    #         xy = (x, y), xytext = (-25, -30),
    #         textcoords = 'offset points', ha = 'right', va = 'bottom',
    #         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
    #         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    #
    # plt.show()
    #
    # gmap = gmplot.GoogleMapPlotter(rep_points['lat'][0], rep_points['lon'][0], 11)
    # gmap.plot(df_min.lat, df_min.lng)
    # gmap.heatmap(rep_points['lat'][:3], rep_points['lon'][:3], radius=20)
    # gmap.draw("user001_work_home.html")
get_user_data()



# generate trajectory
def gen_trajectory(start, destination):
    # Creating a new file:
    # --------------------

    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Create points:
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1234, 5.1234, elevation=1234))
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1235, 5.1235, elevation=1235))
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1236, 5.1236, elevation=1236))

    # You can add routes and waypoints, too...

    print('Created GPX:', gpx.to_xml())


def gen_super_increasing_sequence(current_list):
    sum = 0
    for item in current_list:
        sum += item
    sum += 1
    return sum

def super_increasing_sequence(n):
    start_seq = [AVERAGE_CONTENT_SIZE]
    while n >= 0:
        res = gen_super_increasing_sequence(start_seq)
        start_seq.append(res)
        n -= 1
    return start_seq

b = super_increasing_sequence(20)

#p1 = ()
#p2 = ()
#gen_sentinel_point(p1,p2)