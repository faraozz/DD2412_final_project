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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#for index in range(0, len(x_train)):

def subset_of_data(remove_part, x, y):

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

  print(store_dict)
  print(store_dict_finished)
  return x_new, y_new

x_train, y_train = subset_of_data(0.1, x_train, y_train)



#(x_train, y_train), (x_test, y_test) = stl10.load_data()

#for i in range(0, len(y_train)):
#  y_train[i] = y_train[i]-1

#for i in range(0, len(y_test)):
#  y_test[i] = y_test[i]-1

#print(y_train)

print(f"Total training examples: {len(x_train)}")
print(f"Total test examples: {len(x_test)}")

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 40
CROP_TO = 32
#SEED = 26


PROJECT_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0004





def flip_random_crop(image):
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    return image


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
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
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image


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

# Visualize a few augmented images.
sample_images_one = next(iter(ssl_ds_one))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_one[n].numpy().astype("int"))
    plt.axis("off")
#plt.show()

# Ensure that the different versions of the dataset actually contain
# identical images.
sample_images_two = next(iter(ssl_ds_two))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_two[n].numpy().astype("int"))
    plt.axis("off")
#plt.show()



#i followed the pseudo code from https://arxiv.org/pdf/2006.07733.pdf (original byol paper)


#from our paper:Experiment setup. Unless explicitly stated, in all our experiments, we use ResNet-18 as the backbone network, two-layer
#MLP (with BN and ReLU) as the projector, and a linear predictor. For STL-10 and CIFAR-10, we use SGD as the optimizer
#with learning rate α = 0.03, momentum 0.9, weight decay η¯ = 0.0004 and EMA parameter γa = 0.996. The batchsize is
#128. Each setting is repeated 5 times to compute mean and standard derivation. We report final number as “mean±std”.


N = 2
DEPTH = N * 8 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1



def get_encoder():
    # Input and backbone.
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
  # Input and backbone.
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
    model = tf.keras.Sequential(
        [
            layers.Input((PROJECT_DIM,)),
            layers.Dense(
                PROJECT_DIM,
                use_bias=False,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            ),
            #layers.ReLU(),
            #layers.BatchNormalization(),
            #layers.Dense(PROJECT_DIM),
        ],
        name="predictor",
    )
    return model

def compute_loss(p, z, stopgradient):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    if stopgradient:
      z = tf.stop_gradient(z)

    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

class SimSiam(tf.keras.Model):
    def __init__(self, encoder, predictor, target, useEMA=True, beta=0.9, freq=1, use_direct_pred=True, rho=0.3, epsilon=0.1):
        super(SimSiam, self).__init__()
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
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            #print(z1.shape)


            if self.current_freq%self.freq ==0 and self.use_direct_pred==True:

              F_corr = tfp.stats.correlation(z1, z2)
              #equation 19
              self.F = self.rho*self.F + (1-self.rho)*F_corr

              #print(F.shape)
              e, v = tf.linalg.eigh(self.F)
              #print(e)
              e = tf.linalg.diag(e)
              e_numpy = e.numpy()
              max_e = np.max(e_numpy)
              e = tf.clip_by_value(e, 0., max_e)


              e = tf.math.divide(e, max_e)

              #print(v)
              vT = tf.transpose(v)
              Fest = tf.tensordot(tf.tensordot(v, e, axes=1), vT, axes=1)
              #print(tf.math.equal(F, Fest))
              #print(F)
              #print(Fest)

              #print(max_e)

              p = tf.math.add(tf.math.sqrt(e), self.epsilon)
              p_numpy = p.numpy()
              max_p = np.max(p_numpy)
              p = tf.clip_by_value(p, 1e-4, max_p)
              p_diag = tf.linalg.diag_part(p)
              p_diag = tf.linalg.diag(p_diag)

              new_W_p = tf.tensordot(tf.tensordot(v, p_diag, axes=1), vT, axes=1)


              #now set the weights of the predictor to new_W_p somehow?

              self.predictor.layers[0].set_weights([new_W_p])


              #maybe our self.predictor has bad shape. maybe we should only have 1 layer since it should be linear.
              #also our p_diag is in the wrong form. it should be 2048x2048 not (2048,)
              #also we should send [new_W_p] as the thing to set. thats why its been complaining.
              #also the classses of stl10 seems to be between 1 and 10 instead of 0 and 9


              #todo:
              #also in accuracy evaluation, the classifier wants an object of 32x32x3 which
              #the images of stl10 is not.... how to fix this idk?


              #print(self.predictor.weights.shape)
              # l = 0
              #for layer_predict in self.predictor.layers:
              #  print(l)
              ##  print(layer_predict.trainable_weights)
              #  l = l+1

              #self.predictor.set_weights(new_W_p)

            loss = compute_loss(p1, t2, True) / 2 + compute_loss(p2, t1, True) / 2

        if self.current_freq % self.freq !=0 or self.use_direct_pred == False:

          # Compute gradients and update the parameters.
          learnable_params = (
              self.encoder.trainable_variables + self.predictor.trainable_variables
          )
          gradients = tape.gradient(loss, learnable_params)
          self.optimizer.apply_gradients(zip(gradients, learnable_params))

          if self.useEMA == True:
            self.EMA_updater(self.beta, self.target, self.encoder)

          # Monitor loss.
          self.loss_tracker.update_state(loss)
          self.current_freq += 1
          return {"loss": self.loss_tracker.result()}
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

    #testing with EMA. here we set the weights of the target network to the exponential moving average
    #of the target and encoder network with EMA parameter beta. Now we just need
    #to call this function in self.train_step AFTER encoder has been updated (so that
    #the encoder with the newly updated parameters is used for this update).
    # i used these 2 to check https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
    # https://medium.com/the-dl/easy-self-supervised-learning-with-byol-53b8ad8185d
    #+ some more good links
    #https://keras.io/api/models/model_saving_apis/#get_weights-method
    def EMA_updater(self, beta, target, encoder):

        for layer_target, layer_encoder in zip(self.target.layers, self.encoder.layers):


            if layer_target.trainable_weights != []:
              test = []
              for i in range(len(layer_target.weights)):

                  tensor = tf.math.add(tf.math.scalar_mul(beta, layer_target.weights[i]), tf.math.scalar_mul((1-beta),layer_encoder.weights[i]))
                  test.append(tensor.numpy())

              layer_target.set_weights(test)

# Create a cosine decay learning scheduler.
num_training_samples = len(x_train)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)

# Compile model and start training.
simsiam = SimSiam(get_encoder(), get_predictor(), get_target(), useEMA=True, beta=0.996, freq=1, use_direct_pred=False, rho=0.3, epsilon=0.1)
simsiam.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9), run_eagerly=True)
history = simsiam.fit(ssl_ds, epochs=EPOCHS)

# Visualize the training progress of the model.
plt.plot(history.history["loss"])
plt.grid()
plt.title("Negative Cosine Similairty")
plt.show()

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
    simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output
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
