import cv2

ld = "training/training/"
sd = "training/simplified_training/"

for i in range(1, 1226):
	l = ld+str(i)+".png"
	print("l = ",l)
	im = cv2.imread(l, 0)

	for r in range(im.shape[0]):
		for c in range(im.shape[1]):
			if(im[r][c]>10 and im[r][c]<250):
				im[r][c] = 255
			if(im[r][c]<=10):
				im[r][c] = 0
			if(im[r][c]>=250):
				im[r][c] = 255
	im = cv2.resize(im, (200,200))
	cv2.imwrite(sd+str(i)+".png", im)	