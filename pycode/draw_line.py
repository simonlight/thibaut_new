from PIL import Image, ImageDraw
im = Image.open("2007_000822.jpg") 
width, height = im.size
draw = ImageDraw.Draw(im)
scale=30
w_stride = width/10 
h_stride = height/10
num = (100-scale)/10+1

#for i in range(10):
#	draw.line((((0,0+i*h_stride), (width,0+i*h_stride))), fill=128, width=4)
#	draw.line(((i*w_stride,0), (i*w_stride,height)), fill=128, width=4)
draw.line(((1*w_stride,0), (1*w_stride,height)), fill=128, width=4)
draw.line((((0,0+1*h_stride), (width,0+1*h_stride))), fill=128, width=4)
draw.line(((9*w_stride,0), (9*w_stride,height)), fill=128, width=4)
draw.line((((0,0+9*h_stride), (width,0+9*h_stride))), fill=128, width=4)
#draw.line(((2*w_stride,0), (2*w_stride,height)), fill=128, width=4)
#draw.line((((0,0+2*h_stride), (width,0+2*h_stride))), fill=128, width=4)
#draw.line(((8*w_stride,0), (8*w_stride,height)), fill=128, width=4)
#draw.line((((0,0+8*h_stride), (width,0+8*h_stride))), fill=128, width=4)
#im.show()
im.save("90.png","PNG")

