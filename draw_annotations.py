import cv2

# 8459574585_7bf3d910d6_b.jpg,232,334,680,748,person
# 186,251,382,65
# 8190993390_332a0c0b4f_b.jpg,3,154,338,272
# 6969460448_8758d719d1_b.jpg,219,269,451,574,  602, 819, 2, 1023
# 306.0, 94.0, 590.0, 489.0       574, 639, 448, 513
img = cv2.imread('/Volumes/Seagate Expansion Drive/visual_genome/sg_dataset/sg_train_images/3763808079_6499fd7ccc_b.jpg')

cv2.rectangle(img, (2, 574), (1023, 819), (255,0,0), 2)
#cv2.rectangle(img, (448, 574), (513, 639), (255,0,0), 2)
cv2.imshow('window',img)
cv2.waitKey(0)