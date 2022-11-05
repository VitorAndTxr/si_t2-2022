import csv
import math
import tensorflow as tf

class CongnitiveNavigationDatasnap:
  def __init__(obj):
    obj.Input = []
    obj.Output = []

def generate_X_tensor(ESE45, ESE30, ESF, ESD30, ESD45, EAR):
  ESE45 = float(ESE45)
  ESE30 = float(ESE30)
  ESF = float(ESF)   
  ESD30 = float(ESD30)
  ESD45 = float(ESD45)
  EAR = round(float((EAR/(2*3.1415)+0.5)), 2)

  X =[ [ESD30,ESD45,ESF, EAR],
      [ESD45,ESF,ESE45, EAR],
      [ESE30,ESE45, ESF, EAR]
    ]

  return X

def generate_Y_tensor(SVL, SVA):
  SVL = round(((float(SVL)+1)/2),2)
  SVA = round(((float(SVA)+1)/2),2)

  Y = [SVL,SVA]
    

  return Y 
    
def load_dataset(file):

    dataset = CongnitiveNavigationDatasnap()

    with open(file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            new_input = generate_X_tensor(float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]), float(row[5]))
            new_output = generate_Y_tensor(row[6], row[7])
            dataset.Input.append(new_input)
            dataset.Output.append(new_output)
        
        return dataset

def get_network_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 4)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(18, activation='relu'),
    tf.keras.layers.Dense(18, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu')
    ])
  model.compile(optimizer='adam',
              loss="mean_squared_logarithmic_error",
              metrics=['accuracy'])
  return model
  