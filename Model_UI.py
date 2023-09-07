import streamlit as st
from PIL import Image
import numpy as np
import cv2


# Classes of trafic signs
classes = {
    0 : ' Speed limit (20km/h) ',
    1 : ' Speed limit (30km/h) ',
    2 : ' Speed limit (50km/h) ',
    3 : ' Speed limit (60km/h) ',
    4 : ' Speed limit (70km/h) ',
    5 : ' Speed limit (80km/h) ',
    6 : ' End of speed limit (80km/h) ',
    7 : ' Speed limit (100km/h) ',
    8 : ' Speed limit (120km/h) ',
    9 : ' No passing ',
    10 : ' No passing for vechiles over 3.5 metric tons ',
    11 : ' Right-of-way at the next intersection ',
    12 : ' Priority road ',
    13 : ' Yield ',
    14 : ' Stop ',
    15 : ' No vechiles ',
    16 : ' Vechiles over 3.5 metric tons prohibited ',
    17 : ' No entry ',
    18 : ' General caution ',
    19 : ' Dangerous curve to the left ',
    20 : ' Dangerous curve to the right ',
    21 : ' Double curve ',
    22 : ' Bumpy road ',
    23 : ' Slippery road ',
    24 : ' Road narrows on the right ',
    25 : ' Road work ',
    26 : ' Traffic signals ',
    27 : ' Pedestrians ',
    28 : ' Children crossing ',
    29 : ' Bicycles crossing ',
    30 : ' Beware of ice/snow ',
    31 : ' Wild animals crossing ',
    32 : ' End of all speed and passing limits ',
    33 : ' Turn right ahead ',
    34 : ' Turn left ahead ',
    35 : ' Ahead only ',
    36 : ' Go straight or right ',
    37 : ' Go straight or left ',
    38 : ' Keep right ',
    39 : ' Keep left ',
    40 : ' Roundabout mandatory ',
    41 : ' End of no passing ',
    42 : ' End of no passing by vechiles over 3.5 metric tons ',
    43 : ' Speed limit (5km/h) ',
    44 : ' Speed limit (15km/h) ',
    45 : ' Speed limit (40km/h) ',
    46 : ' Dont Go straight or left ',
    47 : ' Dont Go straight or Right ',
    48 : ' Dont Go straight ',
    49 : ' Dont Go Left ',
    50 : ' Dont Go Left or Right ',
    51 : ' Dont Go Right ',
    52 : ' Dont overtake from Left ',
    53 : ' No Uturn ',
    54 : ' No Car ',
    55 : ' No horn ',
    56 : ' Go straight ',
    57 : ' Go Left ',
    58 : ' Go Left or right ',
    59 : ' Go Right ',
    60 : ' watch out for cars ',
    61 : ' Horn ',
    62 : ' Uturn ',
    63 : ' Road Divider ',
    64 : ' Danger Ahead ',
    65 : ' Zebra Crossing ',
    66 : ' Go right or straight ',
    67 : ' Go left or straight ',
    68 : ' ZigZag Curve ',
    69 : ' Train Crossing ',
    70 : ' Under Construction ',
    71 : ' First aid post ',
    72 : ' Heavy Vehicle Accidents ',
    73 : ' Give Way ',
    74 : ' No stopping ',
    75 : ' Telephone ',
    76 : ' One-way traffic ',
    77 : ' Filling station ',
    78 : ' No vehicles in both directions ',
    79 : ' No entry for cycles ',
    80 : ' No entry for goods vehicles ',
    81 : ' No entry for pedestrians ',
    82 : ' No entry for bullock carts ',
    83 : ' No entry for hand carts ',
    84 : ' No entry for motor vehicles ',
    85 : ' Height limit ',
    86 : ' Weight limit ',
    87 : ' Axle weight limit ',
    88 : ' Length limit ',
    89 : ' No left turn ',
    90 : ' No right turn ',
    91 : ' No overtaking ',
    92 : ' Maximum speed limit (90 km/h) ',
    93 : ' Maximum speed limit (110 km/h) ',
    94 : ' Horn prohibited ',
    95 : ' No parking ',
    96 : ' Turn left ',
    97 : ' Turn right ',
    98 : ' Steep descent ',
    99 : ' Steep ascent ',
    100 : ' Narrow road ',
    101 : ' Narrow bridge ',
    102 : ' Unprotected quay ',
    103 : ' Road hump ',
    104 : ' Dip ',
    105 : ' Loose gravel ',
    106 : ' Falling rocks ',
    107 : ' Cattle ',
    108 : ' Crossroads ',
    109 : ' Side road junction ',
    110 : ' Hotel ',
    111 : ' Oblique side road junction ',
    112 : ' T-junction ',
    113 : ' Y-junction ',
    114 : ' Staggered side road junction ',
    115 : ' Restaurant ',
    116 : ' Guarded level crossing ahead ',
    117 : ' Unguarded level crossing ahead ',
    118 : ' Level crossing countdown marker ',
    119 : ' Parking ',
    120 : ' Bus stop ',
    121 : ' Refreshments ',
}

import keras.models
model = keras.models.load_model('TSR121_model.h5')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    # img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.convert('L')
    image = image.resize((32,32))
    image = np.array(image)
    image = preprocessing(image)
    data.append(image)
    X_test=np.array(data)
    # print(X_test.shape)
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    return image,Y_pred


# Title for the app
st.title("Traffic sign Recognizer")

# Create a file uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
uploaded_file = st.camera_input("Take a picture")
# Check if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_column_width=True, width=100)

    plot,prediction = test_on_img(uploaded_file)
    s = [str(i) for i in prediction] 
    a = int("".join(s)) 

    st.write("Predicted traffic sign is: ", classes[a])
    # print("Predicted traffic sign is: ", classes[a])
