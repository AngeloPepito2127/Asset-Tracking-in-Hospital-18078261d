from difflib import diff_bytes
import math
import json
from pyargus.directionEstimation import *
import numpy as np
import time
import pandas as pd
import json
from datetime import datetime, timedelta
import pickle
from Orange.data.pandas_compat import table_from_frame 
import pandas as pd  
import warnings
import numpy
from sklearn.metrics import multilabel_confusion_matrix
from Anchor import Anchor
warnings.filterwarnings("ignore")
import mysql.connector
from mysql.connector import Error

#These are the credential to access the server used to store the JSON result
host = "your IP"
user = "your username"
pw = "your password"
db = "your database name"

#Initialize DB connection. Return error if connection fail.
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
    except Error as err:
        print(f"Error: '{err}'")

    return connection

#Find the data inside MySQL. Return error if connection fail
def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

#Read the data inside MySql. Return error if fail
def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

def to_plus_minus_pi(angle):
    while angle >= 180:
        angle -= 2 * 180
    while angle < -180:
        angle += 2 * 180
    return angle

#The range of detection is -180 to 180, this function will adjust all angles to this range
def get_angle(X,wavelength):
    # Estimating the spatial correlation matrix
    R = corr_matrix_estimate(X.T, imp="fast")

    array_alignment = np.arange(0, M, 1) * d
    incident_angles = np.arange(-90, 91, 1)
    scanning_vectors = np.zeros((M, np.size(incident_angles)), dtype=complex)
    for i in range(np.size(incident_angles)):
        scanning_vectors[:, i] = np.exp(
            array_alignment * 1j * 2 * np.pi * np.sin(np.radians(incident_angles[i])) / wavelength)  # scanning vector

    ula_scanning_vectors = scanning_vectors

    # Estimate DOA
    MUSIC = DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)
    norm_data = np.divide(np.abs(MUSIC), np.max(np.abs(MUSIC)))
    return float(incident_angles[np.where(norm_data == 1)[0]][0])

#Get the data from MySql and transform them to the right JSON format. It will return the data in form of Dataframe
def getData():
    # datetime object containing current date and time
    
    now = datetime.now()
    past = now - timedelta(seconds=8) #modify to set how long will the data be taken
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    dt_string_past = past.strftime("%Y-%m-%d %H:%M:%S")
  
    q1 = f"""
        SELECT packet from jy_beacon_packet where upd_date between '{dt_string_past}' and '{dt_string}' order by id desc
        """
    connection = create_db_connection(host,user,pw,db)
    results = read_query(connection, q1)

    # Returns a list of lists and then creates a pandas DataFrame
    from_db = []

    for result in results:
        result = list(result)
        from_db.append(result)

    columns = ["packet"]
    df = pd.DataFrame(from_db, columns=columns)

    index = df. index
    number_of_rows = len(index)
    id = {}
    for i in range(number_of_rows):
        dataByte = df.iat[i, 0]
        #print(dataByte)
        data = dataByte.decode('utf-8')
        #print(data)
        result = data.replace("\\", "")
        result1 = result.replace("{\"[{", "[{")
        result2 = result1.replace("\"iq\":\":{\"", "\"iq\":[")
        result3 = result2.replace("\":\"\"}}","]}}]")
        result4 = result3.replace("[{","{")
        result5 = result4.replace("}]","}")
        result6 = result5.replace("},{",",")
        result7 = result6.replace("\"aoa\":{","")
        result8 = result7.replace("]}}","\"}")
        result9 = result8.replace("\"}","]}")
        result10 = result9.replace("]}\":\"","")
        result11 = result10.replace("Z]}","Z\"}")
        res = json.loads(result11)
        id[df.iat[i, 0]] = res

    dataframe = pd.DataFrame(id)
    dataframe = dataframe.transpose()
    result = dataframe.to_json(orient = 'records', compression = 'infer', index = 'true')
    return result

#Loop through all the iq samples to get the phase differences. It will return the phase differences
def msg_loop(messages,wavelength,anchorA, anchorB, anchorC):
    top_left, top_center, top_right, bottom_left, bottom_center, bottom_right = [],[],[],[],[],[]
    left_top, left_center, left_bottom, right_top, right_center, right_bottom = [],[],[],[],[],[]
    total_X, total_Y = [], []
    for i in messages:
        ref_phases = []
        iq_samples = [i['iq'][n:n + 2] for n in range(0, len(i['iq']), 2)] #group 2 by 2

        for iq_idx in range(N_SAMPLES_OF_REF_PERIOD - 1):
            iq_next = complex(iq_samples[iq_idx + 1][0], iq_samples[iq_idx + 1][1])
            iq_cur = complex(iq_samples[iq_idx][0], iq_samples[iq_idx][1])
            phase_next = np.rad2deg(np.arctan2(iq_next.imag, iq_next.real))
            phase_cur = np.rad2deg(np.arctan2(iq_cur.imag, iq_cur.real))
            ref_phases.append((to_plus_minus_pi(phase_next - phase_cur)))
        phase_ref = np.mean(ref_phases)

        iq_2ant_batches = [iq_samples[n:n + 2] for n in range(N_SAMPLES_OF_REF_PERIOD, len(iq_samples),1)]
        for iq_batch_idx, iq_batch in enumerate(iq_2ant_batches[:-1]):
            iq_next = complex(iq_batch[1][0], iq_batch[1][1])
            iq_cur = complex(iq_batch[0][0], iq_batch[0][1])
            phase_next = np.rad2deg(np.arctan2(iq_next.imag, iq_next.real))
            phase_cur = np.rad2deg(np.arctan2(iq_cur.imag, iq_cur.real))
            diff_phase = to_plus_minus_pi((phase_next - phase_cur) - 2*phase_ref)
            if iq_batch_idx % 14 == 0:
                top_left.append(diff_phase) #store elevation phase
            elif iq_batch_idx % 14 == 1:
                top_center.append(diff_phase)
            elif iq_batch_idx % 14 == 2:
                top_right.append(diff_phase)
            elif iq_batch_idx % 14 == 3:
                right_top.append(diff_phase)
            elif iq_batch_idx % 14 == 4:
                right_center.append(diff_phase)
            elif iq_batch_idx % 14 ==5:
                right_bottom.append(diff_phase)
            elif iq_batch_idx % 14 ==7:
                left_top.append(diff_phase)
            elif iq_batch_idx % 14 ==8:
                left_center.append(diff_phase)
            elif iq_batch_idx % 14 ==9:
                left_bottom.append(diff_phase)
            elif iq_batch_idx % 14 ==10:
                bottom_left.append(diff_phase)
            elif iq_batch_idx % 14 ==11:
                bottom_center.append(diff_phase)
            elif iq_batch_idx % 14 == 12:
               bottom_right.append(diff_phase)
            
    top_left_avg = np.average(removeOutliers(top_left, 1.5))
    top_center_avg = np.average(removeOutliers(top_center, 1.5))
    top_right_avg = np.average(removeOutliers(top_right, 1.5))
    bottom_left_avg = np.average(removeOutliers(bottom_left, 1.5))
    bottom_center_avg = np.average(removeOutliers(bottom_center, 1.5))
    bottom_right_avg = np.average(removeOutliers(bottom_right, 1.5))
    left_top_avg = np.average(removeOutliers(left_top, 1.5))
    left_center_avg = np.average(removeOutliers(left_center, 1.5))
    left_bottom_avg = np.average(removeOutliers(left_bottom, 1.5))
    right_top_avg = np.average(removeOutliers(right_top, 1.5))
    right_center_avg = np.average(removeOutliers(right_center, 1.5))
    right_bottom_avg = np.average(removeOutliers(right_bottom, 1.5))

    #limit phase difference reading based on position
    if anchorA.getX() < anchorB.getX():
        if top_left_avg < 20:
            for i in top_left:
                total_X.append(i)
        if top_center_avg  < 20:
            for i in top_center:
                total_X.append(i)
        if  top_right_avg < 20:
            for i in top_right:
                total_X.append(i)
        if bottom_left_avg < 20:
            for i in bottom_left:
                total_X.append(i)
        if bottom_center_avg < 20:
            for i in bottom_center:
                total_X.append(i)
        if bottom_right_avg < 20:
            for i in bottom_right:
                total_X.append(i)
    else: 
        if top_left_avg > -20:
            for i in top_left:
                total_X.append(i)
        if  top_center_avg  > -20:
            for i in top_center:
                total_X.append(i)
        if  top_right_avg > -20 :
            for i in top_right:
                total_X.append(i)
        if bottom_left_avg > -20:
            for i in bottom_left:
                total_X.append(i)
        if bottom_center_avg > -20:
            for i in bottom_center:
                total_X.append(i)
        if bottom_right_avg > -20:
            for i in bottom_right:
                total_X.append(i)


    if anchorA.getY() < anchorC.getY():
        if left_top_avg < 20:
            for i in left_top:
                total_Y.append(i)
        if left_center_avg  < 20:
            for i in left_center:
                total_Y.append(i)
        if left_bottom_avg < 20:
            for i in left_bottom:
                total_Y.append(i)
        if right_top_avg < 20:
            for i in right_top:
                total_Y.append(i)
        if right_center_avg < 20:
            for i in  right_center:
                total_Y.append(i)
        if right_bottom_avg < 20:
            for i in right_bottom:
                total_Y.append(i)
    else: 
        if left_top_avg > -20:
            for i in left_top:
                total_Y.append(i)
        if left_center_avg  > -20:
            for i in left_center:
                total_Y.append(i)
        if left_bottom_avg > -20:
            for i in left_bottom:
                total_Y.append(i)
        if right_top_avg > -20:
            for i in right_top:
                total_Y.append(i)
        if right_center_avg > -20:
            for i in right_center:
                total_Y.append(i)
        if right_bottom_avg > -20:
            for i in right_bottom:
                total_Y.append(i)

    return total_X, total_Y

#Return a string when there is no sample
def no_samples(text):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'{current_time}, {text}, 0 samples')

#Use IQR to remove outliers
def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 60)
    lower_quartile = np.percentile(a, 40)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = []
    for y in a:
        if y >= quartileSet[0] and y <= quartileSet[1]:
            result.append(y)
    return result

#Removing outliers will change the shape of the array, this function is called to add the median of the array so it will return to original size
def addMissingElevation(x_00,phase):
    x0Len = np.size(x_00)
    elLen = len(phase)
    lendif =  x0Len -  elLen
    phases_med = np.median(phase)
    for i in range(0,lendif):
            phase.append(phases_med)
    return phase

#Used to run msg_loop for the selected anchor. It will return final phase angles 
def twofreq(msg_2480,f2480,anchor1,anchor2,anchor3):
    result_2480 = []
    all_x, all_y = [],[]
    if len(msg_2480) > 0:
        result_2480 = msg_loop(msg_2480,f2480,anchor1,anchor2,anchor3)
        a_x = result_2480[0]
        a_y = result_2480[1]

        for i in a_x:
            all_x.append(i)
        for i in a_y:
            all_y.append(i)

        return all_x, all_y
    else:
        return "error"

#prepare the phase differences for MUSIC. It will call get_angle and return the targeted angle
def angle_Music(allpDif,anchorA):
    music_azimuth_phases = removeOutliers(allpDif,1.5)
    azimuth_x_12 = []
    x_total = np.ones(np.size(music_azimuth_phases), dtype=int)
    X = np.zeros((M, np.size(x_total)), dtype=complex)
    X[0, :] = x_total
    for i in music_azimuth_phases:
        azimuth_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = azimuth_x_12
    music_azimuth_angle = 180 - (get_angle(X,wavelength2480) + 90) #change sine to cos
    return music_azimuth_angle

#return the average RSSI
def avgRSSI(rs):
    if len(rs) > 0:
        return np.mean(rs)
    else:

        return rs

#set the angle on each anchor object. It will cll twofreq function   
def settingAngle(msgA,f2480,anchorA,anchorB,anchorC):
    #anchor A is the destination angle
    temp = twofreq(msgA,f2480,anchorA,anchorB,anchorC)
    if temp != "error":
        allpDif_x = temp[0]
        allpDif_y = temp[1]
        if len(allpDif_x)>0 and len(allpDif_y)>0:
            #print(f'allpDif_x: {allpDif_x}')
            music_azimuth_angle = angle_Music(allpDif_x,anchorA)
            #print(f'allpDif_y: {allpDif_y}')
            print(f"{anchorA.getName()} music X: {music_azimuth_angle}")
            music_elevation_angle = angle_Music(allpDif_y,anchorA)
            print(f"{anchorA.getName()} music Y: {music_elevation_angle}")
            anchorA.set_angle_x_mus(music_azimuth_angle)
            anchorA.set_angle_y(music_elevation_angle)
        else:
            print(f'{anchorA.getName()} is error')
    else:
        print(f'{anchorA.getName()} is error')

#Average all the X and Y reading. It will return the final reading of the coordiante
def approx_cor(len0,len1,len2,len3,anchor0, anchor1,anchor2,anchor3):
    #estimated coordinate
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    z = []
    #find X,Z approximation

    #Get X from 0 and 1
    if len0 > 0 and len1 > 0:
            [x1,z1] = getCoordinateX(anchor0,anchor1)
            if z1 > 0:
                z.append(abs(z1))
            print(f'x1: {x1}, z1:{z1}')
            #print(z1)
    else:
        x1 = 0
    #Get X from 2 and 3
    if len2 > 0 and len3 > 0:
            [x2,z2] = getCoordinateX(anchor2,anchor3)
            if z2 > 0:
                z.append(abs(z2))
            print(f'x2: {x2}, z2:{z2}')
            #print(z2)
    else:
        x2 = 0
    
    #Get Y from 0 and 3
    if len0 > 0 and len3 > 0:
            [y1,z3] = getCoordinateY(anchor0,anchor3)
            if z3 > 0:
                z.append(abs(z3))
            #print(z3)
            print(f'y1: {y1}, z3:{z3}')
    else:
        y1 = 0

    #Get Y from 1 and 2
    if (len1 > 0 and len2 > 0):
                [y2,z4] = getCoordinateY(anchor1,anchor2)
                if z4 > 0:
                    z.append(abs(z4))
                #print(z4)
                print(f'y2: {y2}, z4:{z4}')  
    else:
        y2 = 0


    if x1 == 0 and x2 ==0:
        avg_x = "error"
    elif x1 == 0 and x2 != 0:
        avg_x = x2
    elif x2 == 0 and x1 != 0:
        avg_x = x1
    else:
        if abs(x2 - x1) <500:
            avg_x = (x1+x2)/2
        else:
            avg_x = "error"
    
    if y1 == 0 and y2 ==0:
        avg_y = "error"

    elif y1 == 0 and y2 != 0:
        avg_y = y2
    elif y2 == 0 and y1 != 0:
        avg_y = y1
    else:
        if abs(y2 - y1) < 500:
            avg_y = (y1+y2)/2
        else:
            avg_y = "error"

    if len(z) != 0:
        med_z = np.median(z)
        for i in z:
            if abs(i - med_z) > 100:
                i = med_z
        avg_z = np.average(z)
    else:
        avg_z = 0
    
    #print(x1,z1,x2,z2,y1,z3,y2,z4,avg_x,avg_y,avg_z)
    return [avg_x, avg_y, avg_z]

#Triangulate and find the X and Z coordinates.
def getCoordinateX(anchor1,anchor2):

    angle1 = anchor1.get_angle_x_mus()
    angle2 = anchor2.get_angle_x_mus()
    tan_angle1 = math.tan(numpy.deg2rad(angle1))
    tan_angle2 = math.tan(numpy.deg2rad(angle2))
    alpha1 = angle1 + 180
    alpha2 = angle2 + 180
    tan_alpha1 = math.tan(numpy.deg2rad(alpha1))
    tan_alpha2 = math.tan(numpy.deg2rad(alpha2))

    #find coordinate
    if (angle1 == 90 or angle1 == 270):
        x = anchor1.x
        z = anchor2.z - (anchor2.x-anchor1.x)*tan_angle2
    elif (angle2 == 90 or angle2 ==270):
        x = anchor2.x
        z = anchor2.z - (anchor2.x-anchor1.x)*tan_angle1
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >= 0 and alpha2 <= 90):
        #print("between 0 and 90")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >=90 and alpha2 <= 180):
        #print("between 0,90 and 90,180")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 90 and alpha1 <= 180 and alpha2 >=90 and alpha2 <= 180):
        #print("between 90 and 180")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 180 and alpha2 <= 270):
        #print("180,270 ")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 180 and alpha2 <= 270):
        #print("270,360 and 180,270")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 270 and alpha2 <= 360):
        #print("270,360 and 180,270")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 270 and alpha2 <= 360):
        #print("between 90 and 180")
        x = (anchor2.z - anchor1.z +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    else:
        print("err")

    return x,abs(z)

#Triangulate and find the y and Z coordinates. 
def getCoordinateY(anchor1,anchor2):
    angle1 = anchor1.get_angle_y()
    angle2 = anchor2.get_angle_y()
    tan_angle1 = math.tan(numpy.deg2rad(angle1))
    tan_angle2 = math.tan(numpy.deg2rad(angle2))
    alpha1 = angle1 + 180
    alpha2 = angle2 + 180
    tan_alpha1 = math.tan(numpy.deg2rad(alpha1))
    tan_alpha2 = math.tan(numpy.deg2rad(alpha2))

    #find coordinate
    if (angle1 == 90 or angle1 == 270):
        y = anchor1.y
        z = anchor2.z - (anchor2.y-anchor1.y)*tan_angle2
    elif (angle2 == 90 or angle2 ==270):
        y = anchor2.y
        z = anchor2.z - (anchor2.y-anchor1.y)*tan_angle1
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >= 0 and alpha2 <= 90):
        #print("between 0 and 90")
        #y = (anchor1.y*tan_angle1-anchor2.y*tan_angle2)/(tan_angle1 - tan_angle2)
        #z = anchor2.z - (-1)*(((anchor2.y-anchor1.y)*tan_angle1*tan_angle2)/tan_angle1-tan_angle2) 
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_angle1) - (anchor2.y*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >=90 and alpha2 <= 180):
        #print("between 0,90 and 90,180")
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_angle1) - (anchor2.y*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 90 and alpha1 <= 180 and alpha2 >=90 and alpha2 <= 180):
        #print("between 90 and 180")
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_angle1) - (anchor2.y*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 180 and alpha2 <= 270):
        #print("180,270 ")
        #y = (anchor1.y*tan_alpha1-anchor2.y*tan_alpha2)/(tan_alpha1 - tan_alpha2)
        #z = anchor2.z - (-1)*(((anchor2.y-anchor1.y)*tan_alpha1*tan_alpha2)/tan_alpha1-tan_alpha2) 
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_angle1) - (anchor2.y*tan_angle2))/(tan_angle1-tan_angle2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_angle1 - (anchor2.z - anchor1.z))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 180 and alpha2 <= 270):
        #print("270,360 and 180,270")
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_alpha1) - (anchor2.y*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 270 and alpha2 <= 360):
        #print("270,360 and 180,270")
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_alpha1) - (anchor2.y*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 270 and alpha2 <= 360):
        #print("between 90 and 180")
        y = (anchor2.z - anchor1.z +(anchor1.y*tan_alpha1) - (anchor2.y*tan_alpha2))/(tan_alpha1-tan_alpha2)
        z = anchor2.z - ((((anchor2.y-anchor1.y)*tan_alpha1 - (anchor2.z - anchor1.z))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    else:
        print("err")

    return y,abs(z)

NUMBER_MESSAGES = 1 # data accumulation window
SIGMA_FILTER_WINDOW = 5

# antenna array parameters
frequency2480 = 2480
wavelength2480 = 0.12088406
frequency2426 = 2426
wavelength2426 = 0.1235748
frequency2402 = 2402
wavelength2402 = 0.12480952
d = 0.05  # inter element spacing
M = 2  # number of antenna elements in the antenna system
N_SAMPLES_OF_REF_PERIOD = 8

#declare all the anchors position. Adjust as necessary.
anchor0 = Anchor("anchor0",400,0,0)
anchor1 = Anchor("anchor1",0,0,0)
anchor2 = Anchor("anchor2",0,400,0)
anchor3 = Anchor("anchor3",400,400,0)

sign = True
while(sign):
    if __name__ == '__main__':
        errorFlag = 0
        msg0_2480,msg1_2480,msg2_2480,msg3_2480 = [],[],[],[]
        rs0,rs1,rs2,rs3 = [], [], [],[]
        data = json.loads(getData())
        try:
            for i in data[1:]:
                #take the data only for frequency 2480
                if i['frequency'] == frequency2480:
                    #no means the antenna number
                    if i['no'] == 0:
                        rs0.append(i['rssi'])
                        msg0_2480.append(i)
                    elif i['no'] == 1:
                        rs1.append(i['rssi'])
                        msg1_2480.append(i)
                    elif i['no'] == 2:
                        rs2.append(i['rssi'])
                        msg2_2480.append(i)
                    elif i['no'] == 3:
                        rs3.append(i['rssi'])
                        msg3_2480.append(i)
                #take the data only for frequency 2426 as backup
                if i['frequency'] == frequency2426:
                    #no means the antenna number
                    if i['no'] == 0 and len(msg0_2480) < 1:
                        rs0.append(i['rssi'])
                        msg0_2480.append(i)
                    elif i['no'] == 1 and len(msg1_2480) < 1:
                        rs1.append(i['rssi'])
                        msg1_2480.append(i)
                    elif i['no'] == 2 and len(msg2_2480) < 1:
                        rs2.append(i['rssi'])
                        msg2_2480.append(i)
                    elif i['no'] == 3:
                        rs3.append(i['rssi'] and len(msg3_2480) < 1)
                        msg3_2480.append(i)

        except BaseException as e:
                print(e)
                pass

        #store the angles of each locators

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        #make sure none empty
        len0_2480 = len(msg0_2480)   
        len1_2480 = len(msg1_2480)   
        len2_2480 = len(msg2_2480)   
        len3_2480 = len(msg3_2480)   

        if len0_2480 > 0:
            anchor0.set_rssi(avgRSSI(rs0))
            errorFlag = errorFlag +1
        else: 
            print("no data on locator 0")

        if len1_2480 > 0:
            anchor1.set_rssi(avgRSSI(rs1))
            errorFlag = errorFlag +1
        else: 
            print("no data on locator 1")

        if len2_2480 > 0:
            anchor2.set_rssi(avgRSSI(rs2))
            errorFlag = errorFlag +1
        else: 
            print("no data on locator 2")

        if len3_2480 > 0 :
            anchor3.set_rssi(avgRSSI(rs3))
            errorFlag = errorFlag +1
        else: 
            print("no data on locator 3")

    #find x from 1
    if len0_2480 > 0 and len1_2480 >0 and len3_2480 > 0:
        settingAngle(msg0_2480,frequency2480,anchor0,anchor1,anchor3)
    
    if len0_2480 > 0 and len1_2480 >0 and len2_2480 > 0:
        settingAngle(msg1_2480,frequency2480,anchor1,anchor0,anchor2)

    if len3_2480 > 0 and len1_2480 >0 and len2_2480 > 0:
        settingAngle(msg2_2480,frequency2480,anchor2,anchor3,anchor1)

    if len3_2480 > 0 and len0_2480 >0 and len2_2480 > 0:
        settingAngle(msg3_2480,frequency2480,anchor3,anchor2,anchor0)

    if errorFlag > 2:   
        [x,y,z] = approx_cor(len0_2480,len1_2480,len2_2480,len3_2480,anchor0, anchor1,anchor2,anchor3)
        print(f'x,y,z = {x},{y},{z}')
        print()
    else:
        print("not enough locators")
    
    #sign = False #uncomment this to run the program only once
    time.sleep(10) #program will repeat after 10 sec
