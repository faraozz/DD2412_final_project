import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

import resnet_cifar10_v2
import matplotlib.pyplot as plt
from extra_keras_datasets import stl10
from keras.datasets import cifar10
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#globals
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 40
CROP_TO = 32
PROJECT_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0004
N = 2
DEPTH = N * 8 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
dataset = "cifar10"
#dataset = "stl10"

#10% of the original data set used for training.
remove_part_dataset = 0.1

#params to the DirectPred
useEMA=True
beta=0.996
freq=1
use_direct_pred=False
rho=0.3
epsilon=0.1
learning_rate=0.003
momentum=0.9

#params to the linear evaluator
learning_rate=0.0003
decay=0.001


def subset_of_data(remove_part, x, y):
  """
     Removes samples and the corresponding labels from the given samples x and labels y.
     The function removes the same amount of samples from each class.
     remove_part corresponds to the fraction of the original data that should be left.
     i.e. remove_part = 0.1 means that 10% of the original data will be left.

     Returns:
     x_new, y_new
     The new samples with corresponding labels.
  """
  index_array = np.arange(len(x))
  np.random.shuffle(index_array)
  store_dict = np.zeros((1,10))
  store_dict_finished = (int(remove_part*len(x)/10)) * np.ones((1, 10))
  #go through x and y, add them to x_new and y_new
  #only if we have not reached enough samples of that class.
  x_new = []
  y_new = []
  loop_index = 0
  while (store_dict != store_dict_finished).any():

    if store_dict[0, y[index_array[loop_index]][0]] < int(remove_part*len(x)/10):
      store_dict[0, y[index_array[loop_index]][0]]+=1
      x_new.append(x[index_array[loop_index]])
      y_new.append(y[index_array[loop_index]])
    loop_index +=1


  return x_new, y_new



def flip_random_crop(image):
    """
        Flips the image to the left or right and randomly crops the image
        to the size (CROP_TO, CROP_TO, 3)
    """
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    return image


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    """
    Color jitter augmentation of the image.
    """

    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):

    """
    Color drop augmentation of the image.
    """
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    """
    The custom augmentation according to the paper.
    First random flip and crop with p=1. .
    Then color_jitter with p = 0.8
    Then color_drop with p = 0.2 .

    Return the augmented image.
    """
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image




def get_encoder():
    """
    Returns the encoder network. The backbone network is a ResNet16
    and the projector head consists of a two-layer MLP with batch normalization
    and ReLU activation function.
    """
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(
        inputs
    )
    x = resnet_cifar10_v2.stem(x)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")

def get_target():
    """
    Returns the target network. The backbone network is a ResNet16
    and the projector head consists of a two-layer MLP with batch normalization
    and ReLU activation function.
    """
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(
        inputs
    )
    x = resnet_cifar10_v2.stem(x)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="target")

def get_predictor():
    """
        Returns the predictor model. The model consists of a single
        linear layer (with no activation). Is equivalent to a matrix multiplication.

    """
    model = tf.keras.Sequential(
        [
            layers.Input((PROJECT_DIM,)),
            layers.Dense(
                PROJECT_DIM,
                use_bias=False,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            ),

        ],
        name="predictor",
    )
    return model

def compute_loss(p, t, stopgradient):
    """
        Computes the L2-loss according to equation 1 in the report.
        The input p is the output of the predictor
        and the input t is the output of the target network.
    """

    #stop gradient for t as in the architecture.
    if stopgradient:
      t = tf.stop_gradient(t)

    #first we l2 normalize.
    p = tf.math.l2_normalize(p, axis=1)
    t = tf.math.l2_normalize(t, axis=1)

    #which means that minimizing this expression is equivalent to minimizing
    #L2-loss. They do the same in the original paper's authors code.
    return -tf.reduce_mean(tf.reduce_sum((p * t), axis=1))

class DirectPred(tf.keras.Model):

    """
        The DirectPred class. Setting use_direct_pred =False is equivalent to that
        the network is updated only via sgd and thus is a BYOL network.
        Setting useEMA=False also and this network simplifies to a SimSiam network.
    """
    def __init__(self, encoder, predictor, target, useEMA=True, beta=0.9, freq=1, use_direct_pred=True, rho=0.3, epsilon=0.1):
        super(DirectPred, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target = target
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.useEMA = useEMA
        self.beta = beta
        self.current_freq = 0
        self.use_direct_pred = use_direct_pred
        self.freq = freq
        self.rho = rho
        self.epsilon = epsilon
        self.F = np.zeros((PROJECT_DIM, PROJECT_DIM))
    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            t1, t2 = self.target(ds_one), self.target(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)


            #DirectPred update scheme for W_p. See equation 2-4 in the report.
            if self.current_freq%self.freq ==0 and self.use_direct_pred==True:


              #corr(f1, f2)
              F_corr = tfp.stats.correlation(z1, z2)

              #equation 4
              self.F = self.rho*self.F + (1-self.rho)*F_corr

              #Eigen decomposition of F
              sigma, u = tf.linalg.eigh(self.F)

              #sigma
              sigma = tf.linalg.diag(sigma)
              sigma_numpy = sigma.numpy()

              #the maximum eigenvalue of F.
              max_sigma = np.max(sigma_numpy)

              #we make sure theres no negative eigenvalues.
              sigma = tf.clip_by_value(sigma, 0., max_sigma)

              #rescale the eigenvalues by the maximum value so that
              #all the eigenvalues are inbetween 0 and 1.
              sigma = tf.math.divide(sigma, max_sigma)

              #U^T
              uT = tf.transpose(u)

              #p_j see equation 2.
              p = tf.math.add(tf.math.sqrt(sigma), self.epsilon)
              p_numpy = p.numpy()
              max_p = np.max(p_numpy)
              p = tf.clip_by_value(p, 1e-4, max_p)

              #then we make a diagonal matrix with p on the diagonal.
              p_diag = tf.linalg.diag_part(p)
              p_diag = tf.linalg.diag(p_diag)

              #and then we compute the new W_p according to equation 3
              new_W_p = tf.tensordot(tf.tensordot(u, p_diag, axes=1), uT, axes=1)

              #now set the weights of the predictor to new_W_p
              self.predictor.layers[0].set_weights([new_W_p])

            #compute the total loss.
            loss = compute_loss(p1, t2, True) / 2 + compute_loss(p2, t1, True) / 2


        #Computing gradients.

        #if we are not updating via DirectPred then we simply compute gradients
        #via sgd for both the encoder and predictor.
        if self.current_freq % self.freq !=0 or self.use_direct_pred == False:

          # Compute gradients and update the parameters.
          learnable_params = (
              self.encoder.trainable_variables + self.predictor.trainable_variables
          )
          gradients = tape.gradient(loss, learnable_params)
          self.optimizer.apply_gradients(zip(gradients, learnable_params))

          #update the target as exponential moving average with the encoder
          if self.useEMA == True:
            self.EMA_updater(self.beta, self.target, self.encoder)

          # Monitor loss.
          self.loss_tracker.update_state(loss)
          self.current_freq += 1
          return {"loss": self.loss_tracker.result()}

        #if we are updating via directPred then we update the predictor via
        #setting it to W_p which we have already done
        #in the "with tf.GradientTape() as tape: block". Otherwise everything
        #is the same.
        else:
          # Compute gradients and update the parameters.
          learnable_params = (
              self.encoder.trainable_variables #+ self.predictor.trainable_variables
          )
          gradients = tape.gradient(loss, learnable_params)
          self.optimizer.apply_gradients(zip(gradients, learnable_params))


          if self.useEMA == True:
            self.EMA_updater(self.beta, self.target, self.encoder)

          # Monitor loss.
          self.loss_tracker.update_state(loss)
          self.current_freq += 1
          return {"loss": self.loss_tracker.result()}

    #The updater of the target network as an exponential moving average between
    #the target and encoder network.
    def EMA_updater(self, beta, target, encoder):

        for layer_target, layer_encoder in zip(self.target.layers, self.encoder.layers):

            if layer_target.trainable_weights != []:
              new_weights = []
              for i in range(len(layer_target.weights)):

                  tensor = tf.math.add(tf.math.scalar_mul(beta, layer_target.weights[i]), tf.math.scalar_mul((1-beta),layer_encoder.weights[i]))
                  new_weights.append(tensor.numpy())

              layer_target.set_weights(new_weights)


def main():
    global CROP_TO
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        CROP_TO = 32

    elif dataset =="stl10":
        (x_train, y_train), (x_test, y_test) = stl10.load_data()

        #the stl10 datasets classes are between 1 and 10 instead of 0 and 9.
        #so we subtract 1 from all classes.
        for i in range(0, len(y_train)):
          y_train[i] = y_train[i]-1

        for i in range(0, len(y_test)):
          y_test[i] = y_test[i]-1

        #the stl10 images are 96x96x3 size.
        CROP_TO = 96
    else:
        print("error, no such dataset.")

    x_train, y_train = subset_of_data(remove_part_dataset, x_train, y_train)

    print(f"Total training examples: {len(x_train)}")
    print(f"Total test examples: {len(x_test)}")



    # convert data into tensorflow objects
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024)
        .map(custom_augment, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024)
        .map(custom_augment, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))


    # Compile model and start training.
    directPred = DirectPred(get_encoder(), get_predictor(), get_target(), useEMA=True, beta=0.996, freq=1, use_direct_pred=True, rho=0.3, epsilon=0.1)
    directPred.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9), run_eagerly=True)
    history = directPred.fit(ssl_ds, epochs=EPOCHS)



    # We first create labeled `Dataset` objects.
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random resized crops as an augmentation but only to the
    # training set.
    train_ds = (
        train_ds.shuffle(1024)
        .map(lambda x, y: (flip_random_crop(x), y), num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

    # Extract the backbone ResNet20.
    backbone = tf.keras.Model(
        directPred.encoder.input, directPred.encoder.get_layer("backbone_pool").output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = backbone(inputs, training=False)
    outputs = layers.Dense(10, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")

    # Compile model and start training.
    linear_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, decay=0.001),
    )
    history = linear_model.fit(
        train_ds, validation_data=test_ds, epochs=EPOCHS
    )
    _, test_acc = linear_model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))


if __name__ == "__main__":
    main()
