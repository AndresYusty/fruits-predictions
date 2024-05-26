filename = "/content/Fruta/tipos de fruta/papaya/IMG_4153_jpg.rf.543483a14fd6adaab454011dd1567426.jpg"
img = keras.utils.load_img(filename, target_size=image_size)
plt.imshow(img)
from PIL import Image

im = Image.open(filename)

newsize = (180, 180)
im = im.resize(newsize)

np_image = np.array(im)

print(type(np_image), np_image.shape)
img_array = np.expand_dims(np_image,0)
print(type(np_image), img_array.shape)

predictions = model.predict(img_array)
print(predictions)

predicted_class = np.argmax(predictions, axis=1)
score = float(keras.backend.sigmoid(predictions[0][0]))
print(predicted_class)
if predicted_class == 0:
    print(f"This image is {100 * score:.2f}% Mango")
elif predicted_class == 2:
    print(f"This image is {100 * score:.2f}%  banano")
elif predicted_class == 3:
    print(f"This image is {100 * score:.2f}%  limon")
elif predicted_class == 4:
    print(f"This image is {100 * score:.2f}%  maracuya")
elif predicted_class == 5:
    print(f"This image is {100 * score:.2f}%  naranja")
elif predicted_class == 6:
    print(f"This image is {100 * score:.2f}%  papaya")
else:
    print(f"This image is {100 * score:.2f}%  platano")