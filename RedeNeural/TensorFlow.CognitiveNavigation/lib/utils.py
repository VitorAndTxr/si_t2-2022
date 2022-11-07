import csv
import math
import tensorflow as tf
import random


class CongnitiveNavigationDatasnap:
  def __init__(obj):
    obj.Row = []
    obj.Input = []
    obj.Output = []

def mount_training_batch(dataset,batch_size):
  selected_numbers = []
  new_dataset = CongnitiveNavigationDatasnap()

  while len(selected_numbers) < batch_size:
    index = random.randint(0,(len(dataset.Input)-1))
    if(not(index in selected_numbers)):
      new_dataset.Input.append(dataset.Input[index])
      new_dataset.Output.append(dataset.Output[index])
      selected_numbers.append(index)
  
  return new_dataset

def write_csv_row(csv_row,csv_name):
  with open((csv_name+'.csv'),'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(csv_row)
    fd.close()


def generate_X_tensor(ESE45, ESE30, ESF, ESD30, ESD45, EAR):
  ESE45 = float(ESE45)
  ESE30 = float(ESE30)
  ESF = float(ESF)   
  ESD30 = float(ESD30)
  ESD45 = float(ESD45)
  EAR = round(float((EAR)/(3.1415)), 2)

  X =[ [ESD30,ESD45,ESF, EAR],
      [ESD45,ESF,ESE45, EAR],
      [ESE30,ESE45, ESF, EAR]
    ]

  return X

def generate_Y_tensor(SVL, SVA):
  # SVL = round(((float(SVL)+1)/2),2)
  # SVA = round(((float(SVA)+1)/2),2)

  Y = [float(SVL),float(SVA)]
    

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
    tf.keras.layers.Dense(24, activation='tanh'),
    tf.keras.layers.Dense(18, activation='tanh'),
    tf.keras.layers.Dense(2, activation='tanh')
    ])
  # model = tf.keras.models.Sequential()
  # model.add(tf.keras.layers.InputLayer(input_shape =(3,4)))
  # model.add(tf.keras.layers.Dense(6,input_shape =(3,4),activation="tanh"))
  # model.add(tf.keras.layers.Dense(12,input_shape =(3,6),activation="tanh"))
  # model.add(tf.keras.layers.Dense(24,input_shape =(3,12), activation="tanh"))
  # model.add(tf.keras.layers.Dense(12,input_shape =(3,24), activation="tanh"))
  # model.add(tf.keras.layers.Dense(9,input_shape =(3,12), activation="tanh"))
  # model.add(tf.keras.layers.Flatten(input_shape=(3, 9)))
  # model.add(tf.keras.layers.Dense(27, activation="tanh"))
  # model.add(tf.keras.layers.Dense(12, activation="tanh"))
  # model.add(tf.keras.layers.Dense(9, activation="tanh"))
  # model.add(tf.keras.layers.Dense(2, activation="tanh"))

  model.compile(optimizer='adam',
              loss="mean_squared_error",
              metrics=['accuracy'])
  return model
  