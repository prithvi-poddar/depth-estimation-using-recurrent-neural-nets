from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU, Dropout, ConvGRU2D, Conv3D, GRU, UpSampling2D, Permute, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.utils import multi_gpu_model
from data import get_nyu_train_test_data
import numpy as np

autoencoder = Sequential()

# Encoder layers

# E0 Convoutional layer | filter size = (3,3) | stride = 2 | depth = 64 | activation = LeakyRELU | input size = 1242x375
autoencoder.add(TimeDistributed(Conv2D(64, (3, 3),strides = 2 , activation = 'linear'),input_shape = [32, 1242, 375, 3]))
autoencoder.add(LeakyReLU(alpha = 0.1))


# E1 Convolutional GRU | filter size = (3,3) | stride = 2 | depth = 256 | 
autoencoder.add(ConvGRU2D(256, (3,3), strides = 2, input_shape = (None, 32), return_sequences = True))

# E2 Convolutional GRU | filter size = (3,3) | stride = 2 | depth = 512 |
autoencoder.add(ConvGRU2D(512, (3,3), strides = 2, return_sequences = True))

# E3 Convolutional GRU | filter size = (3,3) | stride = 2 | depth = 512 |
autoencoder.add(ConvGRU2D(512, (3,3), strides = 2, return_sequences = True))

# Decoder Layer

# D0 Convolutional layer | filter size = (3,3) | stride = 1 | depth = 512 | activation = LeakyRELU

autoencoder.add(TimeDistributed(Conv2D(512, (3,3), strides = 1, padding = 'same')))
autoencoder.add(LeakyReLU(alpha = 0.1))

# Transpose Reshping
autoencoder.add(Reshape((32, 22, 76, 512)))

# D1 Convolutional GRU | filter size = (3,3) | stride = 1 | depth = 512 |

autoencoder.add(ConvGRU2D(512, (3,3), strides = 1, padding = 'same',return_sequences = True))

# Transpose Reshaping
autoencoder.add(Reshape((32, 76, 22, 512)))

# D2 Convolutional GRU | filter size = (3,3) | stride = 1 | depth = 256 |
autoencoder.add(TimeDistributed(UpSampling2D()))
autoencoder.add(ConvGRU2D(256, (3,3), strides = 1, return_sequences = True))

# Transpose Reshaping
autoencoder.add(Reshape((32, 42, 150, 256)))

# D3 Convolutional GRU | filter size = (3,3) | stride = 1 | depth = 256 |
autoencoder.add(TimeDistributed(UpSampling2D()))
autoencoder.add(ConvGRU2D(256, (3,3), strides = 1, return_sequences = True))

# Transpose Reshaping
autoencoder.add(Reshape((32, 298, 82, 256)))

# D4 Convolutional GRU | filter size = (3,3) | stride = 1 | depth = 128 |
autoencoder.add(TimeDistributed(UpSampling2D()))
autoencoder.add(ConvGRU2D(128, (3,3), strides = 1, return_sequences = False))

# D5 Convolutional layer | filter size = (3,3) | stride = 1 | depth = 3 | activation = tanh
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, (1,1), strides = 1, activation = 'tanh'))

autoencoder.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(horizontal_flip = True)

train = train_datagen.flow_from_directory(
        'data/train',
        target_size=(1242, 375),
        batch_size=32,
        class_mode = 'input')



test_datagen = ImageDataGenerator()

test = train_datagen.flow_from_directory(
        'data/test',
        target_size=(1242, 375),
        batch_size=32,
        class_mode='input')






#train_generator, test_generator = get_nyu_train_test_data( 32 )
autoencoder.fit(train,  validation_data=test, epochs=20, shuffle=False)

