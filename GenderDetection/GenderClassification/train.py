import os 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.callbacks import Callback
from keras.backend import set_session
from keras.optimizers import adam
from keras.optimizers import SGD

#select GPU 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)

classifier = Sequential()

classifier.add(Dense(8, input_dim=128, activation='relu'))
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))  

rate=0.001

classifier.compile(optimizer=adam(learning_rate=rate),  loss='binary_crossentropy', metrics=['accuracy'])

X = numpy.load("X.npy")
Y = numpy.load("Y.npy")


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        # print('Result:'+str(epoch)+ " loss:"+str(round(logs.get('loss'),4))+ " acc:"+str(round(logs.get('accuracy'),4))+ " val_acc:"+str(round(logs.get('val_accuracy'),4)), end='\r', flush=True)
        logs = logs or {}
        loss = logs.get(self.monitor)
        if loss is not None:
            if not self.monitor.endswith("_accuracy"):
                if loss < self.baseline:
                # print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True
            else:
                if loss > self.baseline:
                    # print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True

target_loss=100.0
target_acc=0.0
nb_epoch_train=1000



model_name="train/GenderDetection_v1"

while True :
  callbacks = [TerminateOnBaseline(monitor='accuracy', baseline=target_acc),TerminateOnBaseline(monitor='loss', baseline=target_loss)]
  history=classifier.fit(X,Y, nb_epoch=nb_epoch_train, callbacks=callbacks,batch_size=500 ,verbose=1)
  last_epoch=len(history.history['accuracy'])-1
  acc=history.history['accuracy'][last_epoch]*100.0
  loss=history.history['loss'][last_epoch]
  print("acc             :", round(acc, 2))
  print("target_loss     :", round(loss, 4))

  save_name=model_name+"-"+str(round(loss,4))+"-"+str(round(acc,2))+".h5"
  classifier.save(save_name)
  taget_acc=max(target_acc,history.history['accuracy'][last_epoch])
  target_loss=min(target_loss,loss)



