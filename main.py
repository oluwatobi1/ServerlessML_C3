import ast

import boto3, os, cv2, json, base64
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow import reshape
from io import BytesIO
ROWS = 256
COLS = 256
Channel = 3

def get_modelandweight():
    bucket = boto3.resource("s3").Bucket("reisparcovidbucket")
    # with BytesIO as modelarch:
    #     bucket.download_fileobj(Key="model/efficientnetb6_notop.h5", Fileobj=modelarch)
    #     model = keras.models.load_model(modelarch)
        
    

    base_model = EfficientNetB6(weights='imagenet', include_top=False, input_tensor=Input(shape=(ROWS, COLS, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # # let's add a fully-connected layer
    x = Dense(2024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # The model weights (that are considered the best) are loaded into the model
    #edit model name directory 
    with BytesIO() as weight:  
        bucket.download_file("model/EfficientNetB6.h5", "weight.h5")
        model.load_weights("weight.h5")
        os.remove("weight.h5")


    return model

def predict(event):    
    name = event['body']['name']
    if os.path.exists(os.path.join('', name["body"])):
        image = os.path.join('', name["body"])
        print(name["body"], ' >... event body Found')
    # image = data['file']
    model = get_modelandweight()
    img = cv2.imread(image, cv2.IMREAD_COLOR)       #cv2.IMREAD_GRAYSCALE
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    image = reshape(img,(-1, ROWS, COLS,3))
    result = model.predict(image)
    pred = round(max(result[0][0:2]),2)
    return pred.tolist()


def lambda_handler(event, context):
    print(event)
    ans = predict(event)
    return {'statusCode':200, 
            'body':ans}