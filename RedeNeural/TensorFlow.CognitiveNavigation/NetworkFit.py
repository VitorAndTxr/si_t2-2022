from lib.utils import load_dataset, get_network_model, mount_training_batch, write_csv_row
import tensorflow as tf
import os

def get_model(model_name):
    if os.path.exists(model_name):
        return tf.keras.models.load_model(model_name)
    else:
        return get_network_model()

training_batch = load_dataset('dataset/treino_completo.csv')
epoch_size = 5
training_batch_size = 60
model_name = "modelo_simples_tanh_n_normalizado_2"

teste_final = load_dataset('dataset/teste_completo.csv')
# write_csv_row(['treino_mean_squared_error,treino_accuracy,mean_squared_error,accuracy'],model_name)



model = get_model(model_name)
training_result = model.evaluate(training_batch.Input, training_batch.Output, verbose=0)

teste_result = model.evaluate(teste_final.Input, teste_final.Output, verbose=0)

write_csv_row([str(teste_result[0])+","+str(teste_result[1])+","+str(training_result[0])+","+str(training_result[1])],model_name)


lower_loss = 1
teste_accuracy = 0
training_accuracy = 0

training_model_loss = 1


while teste_accuracy < 0.95 and lower_loss > 0.01 and training_accuracy < 0.98:

    model.fit(
        training_batch.Input, 
        training_batch.Output, 
        epochs=epoch_size,
        verbose=0,
        validation_data=(teste_final.Input, teste_final.Output))

    training_result = model.evaluate(training_batch.Input, training_batch.Output, verbose=1)
    training_accuracy = float(training_result[1])

    teste_result = model.evaluate(teste_final.Input, teste_final.Output, verbose=1)

    lower_loss = float(training_result[0])
    teste_accuracy = float(teste_result[1])
    write_csv_row([str(teste_result[0])+","+str(teste_result[1])+","+str(training_result[0])+","+str(training_result[1])],model_name)
    
model.save(model_name)

print("")







