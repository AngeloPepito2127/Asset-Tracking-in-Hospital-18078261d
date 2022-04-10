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

#The range of detection is -180 to 180, this function will adjust all angles to this range
def to_plus_minus_pi(angle):
    while angle >= 180:
        angle -= 2 * 180
    while angle < -180:
        angle += 2 * 180
    return angle

#This function performs MUSIC and return the detected Angle
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
def msg_loop(messages,wavelength,anchorA, anchorB):
    top_left, top_center, top_right, bottom_left, bottom_center, bottom_right = [],[],[],[],[],[]
    total, top_left_avg, top_center_avg, top_right_avg, bottom_left_avg, bottom_center_avg, bottom_right_avg = [], [], [], [], [], [], []
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

        iq_2ant_batches = [iq_samples[n:n + 4] for n in range(N_SAMPLES_OF_REF_PERIOD, len(iq_samples), 2)]
        for iq_batch_idx, iq_batch in enumerate(iq_2ant_batches[:-1]):

            iq_next = complex(iq_batch[1][0], iq_batch[1][1])
            iq_cur = complex(iq_batch[0][0], iq_batch[0][1])
            phase_next = np.rad2deg(np.arctan2(iq_next.imag, iq_next.real))
            phase_cur = np.rad2deg(np.arctan2(iq_cur.imag, iq_cur.real))
            diff_phase = to_plus_minus_pi((phase_next - phase_cur) - 2*phase_ref)
            if iq_batch_idx % 6 == 0:
                top_left.append(diff_phase) #store elevation phase
            elif iq_batch_idx % 6 == 1:
                top_center.append(diff_phase)
            elif iq_batch_idx % 6 == 2:
                top_right.append(diff_phase)
            elif iq_batch_idx % 6 == 3:
                bottom_left.append(diff_phase)
            elif iq_batch_idx % 6 == 4:
                bottom_center.append(diff_phase)
            elif iq_batch_idx % 6 ==5:
                bottom_right.append(diff_phase)

    top_left_avg = np.average(removeOutliers(top_left, 1.5))
    top_center_avg = np.average(removeOutliers(top_center, 1.5))
    top_right_avg = np.average(removeOutliers(top_right, 1.5))
    bottom_left_avg = np.average(removeOutliers(bottom_left, 1.5))
    bottom_center_avg = np.average(removeOutliers(bottom_center, 1.5))
    bottom_right_avg = np.average(removeOutliers(bottom_right, 1.5))
    #limit phase difference reading based on position
    if anchorA.getX() < anchorB.getX():
        if top_left_avg < 20:
            for i in top_left:
                total.append(i)
        if top_center_avg  < 20:
            for i in top_center:
                total.append(i)
        if  top_right_avg < 20:
            for i in top_right:
                total.append(i)
        if bottom_left_avg < 20:
            for i in bottom_left:
                total.append(i)
        if bottom_center_avg < 20:
            for i in bottom_center:
                total.append(i)
        if bottom_right_avg < 20:
            for i in bottom_right:
                total.append(i)
    else: 
        if top_left_avg > -20:
            for i in top_left:
                total.append(i)
        if  top_center_avg  > -20:
            for i in top_center:
                total.append(i)
        if  top_right_avg > -20 :
            for i in top_right:
                total.append(i)
        if bottom_left_avg > -20:
            for i in bottom_left:
                total.append(i)
        if bottom_center_avg > -20:
            for i in bottom_center:
                total.append(i)
        if bottom_right_avg > -20:
            for i in bottom_right:
                total.append(i)


    return total, top_left_avg, top_center_avg, top_right_avg, bottom_left_avg, bottom_center_avg, bottom_right_avg 

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

#Load the model inside. It will return the model as variable and the format of the dataframe matching the model
def includeModel():
    with open("NeuralNetwork.pkcls", "rb") as f: #you can modify the file location
        model = pickle.load(f)

    #this part is df depend purely on model format
    data= {'2480 top_left': [0], '2480 top_center': [0], '2480 top_right': [0], '2480 bottom_left': [0],'2480 bottom_center': [0],'2480 bottom_right': [0], '2426 top_left': [0], '2426 top_center': [0], '2426 top_right': [0], '2426 bottom_left': [0],'2426 bottom_center': [0],'2426 bottom_right': [0]}  
    df = pd.DataFrame(data)
    angle_data = table_from_frame(df)

    return model, angle_data

#Input the result of each phase differences arrays into the dataframe. It will return the dataframe with correct format and content
def get_angle_model(top_left_2480,top_center_2480,top_right_2480, bottom_left_2480, bottom_center_2480, bottom_right_2480,top_left_2426,top_center_2426,top_right_2426, bottom_left_2426, bottom_center_2426, bottom_right_2426,model,angle_data):
    # Create DataFrame  
    angle_data[0] = [top_left_2480,top_center_2480,top_right_2480, bottom_left_2480, bottom_center_2480, bottom_right_2480,top_left_2426,top_center_2426,top_right_2426, bottom_left_2426, bottom_center_2426, bottom_right_2426]
    result = model(angle_data)

    return result[0]

#Used to run msg_loop for the selected anchor. It will return final phase angles 
def twofreq(msg_2480,msg_2426,f2480,f2426,anchor1,anchor2):
    result_2480, result_2426 = [],[]
    top_left_2480,top_center_2480,top_right_2480, bottom_left_2480, bottom_center_2480, bottom_right_2480 = [], [], [], [], [], []
    top_left_2426,top_center_2426,top_right_2426, bottom_left_2426, bottom_center_2426, bottom_right_2426 = [], [], [], [], [], []
    all = []
    if len(msg_2480) > 0:
        result_2480 = msg_loop(msg_2480,f2480,anchor1,anchor2)
        a_anchor_2480 = result_2480[0]
        top_left_2480 = result_2480[1]
        top_center_2480 = result_2480[2]
        top_right_2480 = result_2480[3]
        bottom_left_2480 = result_2480[4]
        bottom_center_2480 = result_2480[5]
        bottom_right_2480 = result_2480[6]
        for i in a_anchor_2480:
            all.append(i)
        print("")
    if len(msg_2426) > 0:
        result_2426 = msg_loop(msg_2426,f2426,anchor1,anchor2)
        a_anchor_2426 = result_2426[0]
        top_left_2426 = result_2426[1]
        top_center_2426 = result_2426[2]
        top_right_2426 = result_2426[3]
        bottom_left_2426 = result_2426[4]
        bottom_center_2426 = result_2426[5]
        bottom_right_2426 = result_2426[6]
        if len(msg_2480) == 0:
            for i in a_anchor_2426:
                all.append(i)
        print("") 
    if len(msg_2480) > 0 and len(msg_2426) > 0:
        model_angle = get_angle_model(top_left_2480,top_center_2480,top_right_2480, bottom_left_2480, bottom_center_2480, bottom_right_2480,top_left_2426,top_center_2426,top_right_2426, bottom_left_2426, bottom_center_2426, bottom_right_2426,model,angle_df)
        print(f"{anchor1.getName()} model: {model_angle}")
        return all, model_angle
    else:
        return "error"

#prepare the phase differences for MUSIC. It will call get_angle and return the targeted angle
def angle_Music(allpDif,anchorA):
    music_azimuth_phases = removeOutliers(allpDif,1.5)
    #print(music_azimuth_phases )
    azimuth_x_12 = []
    x_total = np.ones(np.size(music_azimuth_phases), dtype=int)
    X = np.zeros((M, np.size(x_total)), dtype=complex)
    X[0, :] = x_total
    for i in music_azimuth_phases:
        azimuth_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = azimuth_x_12
    music_azimuth_angle = 180 - (get_angle(X,wavelength2480) + 90) #change sine to cos
    print(f"{anchorA.getName()} music: {music_azimuth_angle}")
    return music_azimuth_angle

#originally to compare which method works better.
def whichMethod(model, music, anchorA):
    print(f'{anchorA.getName()}: {music}')
    return music

#compare which RSSI is the strongest. It will return locators with the first and second strongest RSSI reading
def compareRSSI(rs0,rs1,rs2,rs3):
    fullRS = [rs0,rs1,rs2,rs3]
    rsX = fullRS[0].get_rssi()
    rsY = fullRS[1].get_rssi()
    rsZ = fullRS[2].get_rssi()
    rsM = fullRS[3].get_rssi()
    temp = [rsX, rsY, rsZ, rsM]
    index_max = max(range(len(temp)), key=temp.__getitem__)
    maxRS = fullRS[index_max]
    fullRS.pop(index_max) 
    rsX = fullRS[0].get_rssi()
    rsY = fullRS[1].get_rssi()
    rsZ = fullRS[2].get_rssi()
    temp = [rsX, rsY, rsZ]
    index_max = max(range(len(temp)), key=temp.__getitem__)
    maxRS2 = fullRS[index_max]
    return maxRS, maxRS2

#Triangulate and find the coordinate using MUSIC
def getCoordinateX_mod(anchor1,anchor2):

    angle1 = anchor1.get_angle_x_mod()

    angle2 = anchor2.get_angle_x_mod()
    if anchor1.getX() < anchor2.getX():
        if angle1 < 90:
            angle1 = 90
        if angle2 > 90:
            angle2 = 90
    else:
        if angle1 > 90:
            angle1 = 90
        if angle2 < 90:
            angle2 = 90
    tan_angle1 = math.tan(numpy.deg2rad(angle1))
    tan_angle2 = math.tan(numpy.deg2rad(angle2))
    alpha1 = angle1 + 180
    alpha2 = angle2 + 180
    tan_alpha1 = math.tan(numpy.deg2rad(alpha1))
    tan_alpha2 = math.tan(numpy.deg2rad(alpha2))

    #find coordinate
    if (angle1 == 90 or angle1 == 270):
        x = anchor1.x
        y = anchor2.y - (anchor2.x-anchor1.x)*tan_angle2
    elif (angle2 == 90 or angle2 ==270):
        x = anchor2.x
        y = anchor2.y - (anchor2.x-anchor1.x)*tan_angle1
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >= 0 and alpha2 <= 90):
        #print("between 0 and 90")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >=90 and alpha2 <= 180):
        #print("between 0,90 and 90,180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 90 and alpha1 <= 180 and alpha2 >=90 and alpha2 <= 180):
        #print("between 90 and 180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 180 and alpha2 <= 270):
        #print("180,270 ")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 180 and alpha2 <= 270):
        #print("270,360 and 180,270")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 270 and alpha2 <= 360):
        #print("270,360 and 180,270")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 270 and alpha2 <= 360):
        #print("between 90 and 180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    else:
        print("err")

    return x,-y

#Triangulate and find the coordinate using Model
def getCoordinateX_mus(anchor1,anchor2):

    angle1 = anchor1.get_angle_x_mus()
    angle2 = anchor2.get_angle_x_mus()
    if anchor1.getX() < anchor2.getX():
        if angle1 < 90:
            angle1 = 90
        if angle2 > 90:
            angle2 = 90
    else:
        if angle1 > 90:
            angle1 = 90
        if angle2 < 90:
            angle2 = 90
    tan_angle1 = math.tan(numpy.deg2rad(angle1))
    tan_angle2 = math.tan(numpy.deg2rad(angle2))
    alpha1 = angle1 + 180
    alpha2 = angle2 + 180
    tan_alpha1 = math.tan(numpy.deg2rad(alpha1))
    tan_alpha2 = math.tan(numpy.deg2rad(alpha2))

    #find coordinate
    if (angle1 == 90 or angle1 == 270):
        x = anchor1.x
        y = anchor2.y - (anchor2.x-anchor1.x)*tan_angle2
    elif (angle2 == 90 or angle2 ==270):
        x = anchor2.x
        y = anchor2.y - (anchor2.x-anchor1.x)*tan_angle1
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >= 0 and alpha2 <= 90):
        #print("between 0 and 90")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2)
    elif (alpha1>= 0 and alpha1 <= 90 and alpha2 >=90 and alpha2 <= 180):
        #print("between 0,90 and 90,180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 90 and alpha1 <= 180 and alpha2 >=90 and alpha2 <= 180):
        #print("between 90 and 180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_angle1) - (anchor2.x*tan_angle2))/(tan_angle1-tan_angle2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_angle1 - (anchor2.y - anchor1.y))/(tan_angle1-tan_angle2))*tan_angle2) 
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 180 and alpha2 <= 270):
        #print("180,270 ")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 180 and alpha2 <= 270):
        #print("270,360 and 180,270")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 180 and alpha1 <= 270 and alpha2 >= 270 and alpha2 <= 360):
        #print("270,360 and 180,270")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    elif (alpha1>= 270 and alpha1 <= 360 and alpha2 >= 270 and alpha2 <= 360):
        #print("between 90 and 180")
        x = (anchor2.y - anchor1.y +(anchor1.x*tan_alpha1) - (anchor2.x*tan_alpha2))/(tan_alpha1-tan_alpha2)
        y = anchor2.y - ((((anchor2.x-anchor1.x)*tan_alpha1 - (anchor2.y - anchor1.y))/(tan_alpha1-tan_alpha2))*tan_alpha2)
    else:
        print("err")

    return x,abs(y)

#return the average RSSI
def avgRSSI(rs):
    if len(rs) > 0:
        return np.mean(rs)
    else:
        return rs
    
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

#this set the model used
start_model = includeModel()
model = start_model[0]
angle_df = start_model[1]

#declare all the anchors position. Adjust as necessary.
anchor0 = Anchor("anchor0",400,0,95)
anchor1 = Anchor("anchor1",200,0,95)
anchor2 = Anchor("anchor2",0,0,95)
anchor3 = Anchor("anchor3",0,0,95)

sign = True
while(sign):
    if __name__ == '__main__':
        errorFlag = 0
        msg0_2480,msg1_2480,msg2_2480,msg3_2480 = [],[],[],[]
        msg0_2426,msg1_2426,msg2_2426,msg3_2426 = [],[],[],[]
        l0a, l1a, l2a,l3a = [], [], [], []
        rs0,rs1,rs2,rs3 = [], [], [],[]
        data = json.loads(getData())
        try:
            #take the data only for frequency 2480
            for i in data[1:]:
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
            #take the data only for frequency 2426
                if i['frequency'] == frequency2426:
                    #no means the antenna number
                    if i['no'] == 0:
                        rs0.append(i['rssi'])
                        msg0_2426.append(i)
                    elif i['no'] == 1:
                        rs1.append(i['rssi'])
                        msg1_2426.append(i)
                    elif i['no'] == 2:
                        rs2.append(i['rssi'])
                        msg2_2426.append(i)
                    elif i['no'] == 3:
                        rs3.append(i['rssi'])
                        msg3_2426.append(i)
        except BaseException as e:
                print(e)
                pass

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        #make sure none of the data are empty
        len0_2480 = len(msg0_2480)   
        len0_2426 = len(msg0_2426) 
        len1_2480 = len(msg1_2480)   
        len1_2426 = len(msg1_2426)
        len2_2480 = len(msg2_2480)   
        len2_2426 = len(msg2_2426) 
        len3_2480 = len(msg3_2480)   
        len3_2426 = len(msg3_2426) 

        if len0_2480 > 0 or len0_2426 > 0:
            anchor0.set_rssi(avgRSSI(rs0))
            errorFlag = errorFlag +1
            print(f'rs0: {avgRSSI(rs0)}')
        else: 
            print("no data on locator 0")

        if len1_2480 > 0 or len1_2426 > 0:
            anchor1.set_rssi(avgRSSI(rs1))
            print(f'rs1: {avgRSSI(rs1)}')
            errorFlag = errorFlag +1
        else: 
            print("no data on locator 1")

        if len2_2480 > 0 or len2_2426 > 0:
            anchor2.set_rssi(avgRSSI(rs2))
            errorFlag = errorFlag +1
            print(f'rs2: {avgRSSI(rs2)}')
        else: 
            print("no data on locator 2")

        if len3_2480 > 0 or len3_2426 > 0:
            anchor3.set_rssi(avgRSSI(rs3))
            errorFlag = errorFlag +1
            print(f'rs3: {avgRSSI(rs3)}')
        else: 
            print("no data on locator 3")

        #only do the following when there are enough data    
        if errorFlag > 1:    
            usedLocators = compareRSSI(anchor0,anchor1,anchor2,anchor3)

            msgA_2480, msgA_2426, msgB_2480, msgB_2426,anchorA, anchorB = [],[],[],[],[],[]
            if usedLocators[0].getName() == "anchor0":
                msgA_2480 = msg0_2480
                msgA_2426 = msg0_2426
                anchorA = anchor0
            elif usedLocators[0].getName() == "anchor1":
                msgA_2480 = msg1_2480
                msgA_2426 = msg1_2426
                anchorA = anchor1
            elif usedLocators[0].getName() == "anchor2":
                msgA_2480 = msg2_2480
                msgA_2426 = msg2_2426
                anchorA = anchor2
            else:
                msgA_2480 = msg3_2480
                msgA_2426 = msg3_2426
                anchorA = anchor3

            if usedLocators[1].getName() == "anchor0":
                msgB_2480 = msg0_2480
                msgB_2426 = msg0_2426
                anchorB = anchor0
            elif usedLocators[1].getName() == "anchor1":
                msgB_2480 = msg1_2480
                msgB_2426 = msg1_2426
                anchorB = anchor1
            elif usedLocators[1].getName() == "anchor2":
                msgB_2480 = msg2_2480
                msgB_2426 = msg2_2426
                anchorB = anchor2
            else:
                msgB_2480 = msg3_2480
                msgB_2426 = msg3_2426
                anchorB = anchor3

            #Find the angle of first anchor
            temp = twofreq(msgA_2480,msgA_2426,frequency2480,frequency2426,anchorA,anchorB)
            if temp != "error":
                allpDif = temp[0]
                model_azimuth_angle = temp[1]
                anchorA.set_angle_x_mod(model_azimuth_angle)
                music_azimuth_angle = angle_Music(allpDif,anchorA)
                anchorA.set_angle_x_mus(music_azimuth_angle)
                l0a = whichMethod(model_azimuth_angle, music_azimuth_angle,anchorA)
            else:
                errorFlag = errorFlag - 1
                print("not enough data from one frequency")

            #Find the angle of second anchor
            temp = twofreq(msgB_2480,msgB_2426,frequency2480,frequency2426,anchorB,anchorA)
            if temp != "error":
                allpDif = temp[0]
                model_azimuth_angle = temp[1]
                anchorB.set_angle_x_mod(model_azimuth_angle)
                music_azimuth_angle = angle_Music(allpDif,anchorB)
                anchorB.set_angle_x_mus(music_azimuth_angle)
                l0a = whichMethod(model_azimuth_angle, music_azimuth_angle,anchorB)
            else:
                errorFlag = errorFlag - 1
                print("not enough data from one frequency")
            

            if errorFlag > 1:   
                [x,y] = getCoordinateX_mod(usedLocators[0], usedLocators[1])
                print(f'{x},{y}')
                [x,y] = getCoordinateX_mus(usedLocators[0], usedLocators[1])
                print(f'{x},{y}')
            else:
                print("not enough locators")
        else:
            print("not enough locators")
    
    #sign = False #uncomment this to run the program only once
    time.sleep(10) #program will repeat after 10 sec
