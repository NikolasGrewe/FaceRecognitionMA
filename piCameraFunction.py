try:
    import os
    
    def takePicture():
        os.system('libcamera-jpeg --ev 0.5 -o picture.jpg --width 250 --height 250')

except:
    print("An Error has occurred utilizing the camera. Make sure your computer supports libcamera/has a camera attached. ")
