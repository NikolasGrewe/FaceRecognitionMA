try:
    from time import sleep
    from picamera import PiCamera

    camera = PiCamera()
    camera.resolution = (2592, 1944)
    camera.start_preview()
    
    def takePicture(filename):
        sleep(3)
        camera.capture('facePhoto.jpg')

except:
    print("An Error has occurred initializing the PiCamera")
