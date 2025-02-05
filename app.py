from vision import Vision
import cv2 as cv

if __name__ == "__main__":
    v = Vision()
    draw_debug = False 
    while True:

        key = cv.waitKey(10)
        if key == 27:  # escape key
            break
        
        if key == 46: # . key
            draw_debug = not draw_debug 

        stop, hand_id = v.run_frame(draw_debug)
        if stop:
            break