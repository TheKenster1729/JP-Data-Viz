from PIL import Image

# open images
img1 = Image.open("/Users/kcox1729/Downloads/Screenshot_20250923-144931.png")
img2 = Image.open("/Users/kcox1729/Downloads/PXL_20250923_184927985.MP.jpg")

# make sure they are the same height
h = max(img1.height, img2.height)
new_img = Image.new("RGB", (img1.width + img2.width, h))

# paste side by side
new_img.paste(img1, (0, 0))
new_img.paste(img2, (img1.width, 0))

new_img.save("/Users/kcox1729/Downloads/combined.jpg")
