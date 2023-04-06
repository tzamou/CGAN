import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense,LeakyReLU,Activation,\
            Conv2D,Conv2DTranspose,Flatten,Reshape,BatchNormalization,Embedding,multiply
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam
import os
try:
    from tensorflow.keras.utils import plot_model
except Exception as e:
    print(e)

(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

class CGAN:
    def __init__(self,gen_lr=0.0002,dis_lr=0.0002,model_name=None):
        if model_name == None:
            self.modelname = self.__class__.__name__
        else:
            self.modelname = model_name
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.genModel = self.build_generator()
        self.disModel = self.build_discriminator()
        self.advModel = self.build_adversialmodel()
        if not os.path.isdir('./result'):
            os.mkdir('./result')
    def generator_block(self,input,u,k,s=2,p='same',a='leakyrelu',bn=True):
        x = Conv2DTranspose(u,kernel_size=k,strides=s,padding=p)(input)
        if a=='leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif a=='sigmoid':
            x = Activation('sigmoid')(x)
        elif a=='tanh':
            x = Activation('tanh')(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    def discriminator_block(self,input,u,k,s=2,p='padding',a='leakyrelu'):
        x = Conv2D(u,kernel_size=k,strides=s,padding=p)(input)
        if a=='leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif a=='sigmoid':
            x = Activation('sigmoid')(x)
        return x
    def build_generator(self):
        noise_input_layer = Input(shape=(100, ))
        label_input_layer = Input(shape=(1, ),dtype=np.int32)
        label_embedding = Flatten()(Embedding(10,100)(label_input_layer))
        input_layer = multiply([noise_input_layer,label_embedding])
        x = Dense(7*7*32)(input_layer)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape((7, 7, 32))(x)
        x = self.generator_block(x, u=128, k=2, s=2, p='same')
        x = self.generator_block(x, u=256, k=2, s=2, p='same')
        out = self.generator_block(x,u=1,k=1,s=1,p='same',a='sigmoid',bn=False)

        opt=Adam(learning_rate=self.gen_lr,beta_1=0.5)
        model = Model(inputs=[noise_input_layer,label_input_layer],outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt)
        plot_model(model,to_file=f'./result/{self.modelname}_gen.png',show_shapes=True)
        return model

    def build_discriminator(self):
        img_input_layer = Input(shape=(28, 28, 1))
        label_input_layer = Input(shape=(1,), dtype=np.int32)
        label_embedding = Flatten()(Embedding(10, 28*28)(label_input_layer))
        label_embedding = Reshape(target_shape=(28,28))(label_embedding)
        input_layer = multiply([img_input_layer, label_embedding])
        x = self.discriminator_block(input=input_layer,u=256,k=2,s=2,p='same',a='leakyrelu')
        x = self.discriminator_block(input=x, u=128, k=2, s=2, p='same', a='leakyrelu')
        x = Flatten()(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        out = Activation('sigmoid')(x)

        opt = Adam(learning_rate=self.dis_lr,beta_1=0.5)
        model = Model(inputs=[img_input_layer,label_input_layer],outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        plot_model(model, to_file=f'./result/{self.modelname}_dis.png', show_shapes=True)
        return model

    def build_adversialmodel(self):
        z_noise_input = Input(shape=(100, ))
        label_input_layer = Input(shape=(1,), dtype=np.int32)
        gen_sample = self.genModel([z_noise_input,label_input_layer])
        self.disModel.trainable = False
        out = self.disModel([gen_sample,label_input_layer])
        opt = Adam(learning_rate=self.gen_lr,beta_1=0.5)
        model = Model(inputs=[z_noise_input,label_input_layer], outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt)
        return model
    def train(self,epochs=30000):
        batch_size = 32
        dloss = []
        gloss = []
        for i in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            noise = np.random.normal(0, 1, [batch_size, 100])
            real_label = y_train[idx]
            fake_images = self.genModel([noise,real_label])
            real_images = x_train[idx]


            y_real = np.ones((batch_size, 1))
            y_fake = np.zeros((batch_size, 1))

            d_loss_fake = self.disModel.train_on_batch(x=[fake_images,real_label], y=y_fake)
            d_loss_real = self.disModel.train_on_batch(x=[real_images,real_label], y=y_real)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            dloss.append(d_loss[0])
            noise = np.random.normal(0, 1, [batch_size, 100])
            g_loss = self.advModel.train_on_batch([noise,real_label], y_real)
            g_loss2 = self.advModel.train_on_batch([noise,real_label], y_real)
            g_loss3 = self.advModel.train_on_batch([noise,real_label], y_real)
            gloss.append((g_loss+g_loss2+g_loss3)/3)
            print(f"epochs:{i+1} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.3f}] [G loss: {g_loss}]")
            if (i+1)%500==0 or i==0:
                num_images=16
                noise = np.random.normal(0,1,[num_images, 100])
                idx = np.random.randint(0, x_train.shape[0], num_images)
                real_label = y_train[idx]
                generated_images = self.genModel([noise,real_label])
                # generated_images = 0.5 * generated_images + 0.5
                plt.figure(figsize=(8, 8))
                plt.title(f'{i+1} epochs train')
                for j in range(num_images):
                    plt.subplot(4, 4, j + 1)
                    plt.imshow(generated_images[j, :, :, 0], cmap="gray")
                    plt.axis("off")
                plt.savefig(f"images/epoch{i+1}.png")

        self.genModel.save(f'./result/{self.modelname}_generator2.h5')
        self.disModel.save(f'./result/{self.modelname}_discriminator2.h5')
        self.advModel.save(f'./result/{self.modelname}_adversarial2.h5')
        plt.clf()
        # plt.ylim(-0.1,1)
        plt.title('loss')
        plt.plot(gloss,label='g')
        plt.plot(dloss,label='d')
        np.save('./result/gloss.npy', arr=gloss)
        np.save('./result/dloss.npy', arr=dloss)
        plt.legend(loc='best')
        plt.savefig(f"./result/loss.png")

    def predict(self,num_images = 25, label=0):
        noise = np.random.normal(0, 1, [num_images, 100])
        if label == -1:
            label = np.random.randint(size=num_images, low=0, high=10)
            print(label)
        else:
            label = np.array([label for _ in range(num_images)], dtype=np.int32)
        # label = np.random.randint(size=num_images,low=0,high=10)
        model = load_model(f'./result/{self.modelname}_generator2.h5')
        generated_images = model([noise,label])
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        plt.suptitle(f'The Images Of {self.modelname} Generate', fontsize=17)
        sub = int(np.sqrt(num_images))
        for j in range(num_images):
            plt.subplot(sub, sub, j + 1)
            plt.title(label[j],fontsize=12)
            plt.imshow(generated_images[j, :, :, 0], cmap="gray")
            plt.axis("off")
        plt.show()

    def showinput(self,num_images = 25):
        noise = np.random.normal(0, 1, [num_images, 100])
        noise = noise.reshape((-1,10,10,1))
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        sub = int(np.sqrt(num_images))
        for j in range(num_images):
            plt.subplot(sub, sub, j + 1)
            plt.imshow(noise[j, :, :, 0], cmap="gray")
            plt.axis("off")
        plt.show()
if __name__=='__main__':
    gan=CGAN(model_name='cGAN')
    # gan.train(epochs=50000)
    # showinput()
    # for i in range(10):
    gan.predict(label=-1)

