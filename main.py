import preprocessing
import network

# convert csv into images
train_images, train_labels = preprocessing.load_data("MNIST-Dataset/sign_mnist_train.csv")
test_images, test_labels = preprocessing.load_data("MNIST-Dataset/sign_mnist_test.csv")

train_images, test_images = train_images/255, test_images/255

# show something to check correctness
preprocessing.show_image(0, train_images, train_labels, preprocessing.letters)

model = network.create_model()
network.compile_model(model)
history = network.train_model(model, train_images, train_labels, test_images, test_labels, epochs=2)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
