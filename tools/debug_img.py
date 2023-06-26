
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap
from PIL import Image

# Load depth data from .npy file
pred_files = glob('/home/wugaoqiang/work/depth/SurroundDepth/data/nuscenes/pred' + "/*.npy")

# def convertPNG(pngfile, outdir):
#     # READ THE DEPTH
#     im_depth = cv2.imread(pngfile)
#     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
#     im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
#     #convert to mat png
#     im=Image.fromarray(im_color)
#     #save image
#     im.save(os.path.join(outdir,os.path.basename(pngfile)))


front_depth_list = []
for path in pred_files:
    filename = os.path.basename(path)
    start_index = filename.rfind('_')
    end_index = filename.rfind('.')
    camera_id = filename[:start_index]
    if camera_id == 'front':
        depth_data = np.load(path)
        front_depth_list.append(depth_data)
        # Normalize depth values between 0 and 1
        # depth_norm = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())

        # # Create a colormap with 256 colors
        # cmap = plt.get_cmap('hot', 256)

        # Map normalized depth values to RGB colors using the colormap
        # depth_color = cmap(depth_norm)
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=15), cv2.COLORMAP_JET)

        # Show the depth map in pseudocolor
        plt.imshow(depth_color)
        plt.show()
        # plt.pause(0.05)
        # plt.close()


        # data = np.load('frame_{}.npy'.format(i))
        # # Create a new plot
        # fig, ax = plt.subplots()

        # # Display the data as an image
        # im = ax.imshow(data, cmap='gray')

        # # Add a colorbar
        # # cbar = ax.figure.colorbar(im, ax=ax)

        # # Set the axis labels
        # # ax.set_xlabel('X Label')
        # # ax.set_ylabel('Y Label')
        # # ax.set_title('Frame {}'.format(i))

        # # Show the plot and pause for a short time to simulate video playback
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close()


# Load the numpy arrays from .npy files
# frame1 = np.load('frame1.npy')
# frame2 = np.load('frame2.npy')
# if len(front_depth_list) == 0:
#     exit()

# # Create a video writer object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter('output.mp4', fourcc, 25, (front_depth_list[0].shape[1], front_depth_list[0].shape[0]))

# # Loop through the frames and display them
# for front_depth in front_depth_list:  # Change 100 to the number of frames you have

#     # Normalize the frame to values between 0 and 255
#     frame_norm = cv2.normalize(front_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#     # Convert the frame to BGR color space and write it to the video file
#     frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
#     writer.write(frame_bgr)

#     # Display the frame
#     cv2.imshow('Frame', frame_norm)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Release the video writer and close all windows
# writer.release()
# cv2.destroyAllWindows()


# import cv2
# import os.path
# import glob
# import numpy as np


# def convertPNG(pngfile, outdir):
#     # READ THE DEPTH
#     im_depth = cv2.imread(pngfile)
#     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
#     im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
#     #convert to mat png
#     im=Image.fromarray(im_color)
#     #save image
#     im.save(os.path.join(outdir,os.path.basename(pngfile)))

# for pngfile in glob.glob("PNG FILE"):#C:/Users/BAMBOO/Desktop/source pics/rgbd_6/depth/*.png
#     convertPNG(pngfile,"TARGET FILE")#C:/Users/BAMBOO/Desktop/source pics/rgbd_6/color