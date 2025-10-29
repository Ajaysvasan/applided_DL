from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray

# ========= Load Image =========
# load image as pixel array (Matplotlib)
data = mpimg.imread('opera_house.jpg')
print("Matplotlib Image dtype:", data.dtype)
print("Matplotlib Image shape:", data.shape)

plt.imshow(data)
plt.title("Original Image (Matplotlib)")
plt.axis('off')
plt.show()

# ========= Resize Image (PIL) =========
pil_image = Image.open('opera_house.jpg')
new_image = pil_image.resize((500, 500))
new_image.save('myimage_500.jpg')

plt.imshow(new_image)
plt.title("Resized Image (500x500)")
plt.axis('off')
plt.show()

# ========= Convert to NumPy Array =========
data_array = asarray(pil_image)
print("NumPy Array Shape:", data_array.shape)

image2 = Image.fromarray(data_array)
print("Image Format:", image2.format)  # might print None if loaded from array
print("Image Mode:", image2.mode)
print("Image Size:", image2.size)

# ========= Flip Image =========
hoz_flip = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
ver_flip = pil_image.transpose(Image.FLIP_TOP_BOTTOM)

plt.figure(figsize=(6, 10))
plt.subplot(3, 1, 1)
plt.imshow(pil_image)
plt.title("Original")
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(hoz_flip)
plt.title("Horizontal Flip")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(ver_flip)
plt.title("Vertical Flip")
plt.axis('off')

plt.show()

# ========= Rotate Image =========
plt.figure(figsize=(6, 10))
plt.subplot(3, 1, 1)
plt.imshow(pil_image)
plt.title("Original")
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(pil_image.rotate(45))
plt.title("Rotated 45°")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(pil_image.rotate(90))
plt.title("Rotated 90°")
plt.axis('off')

plt.show()
