import os, sys, cv2, numpy as np, pyrealsense2 as rs

def create_depth_vis(depth_image):
	min_val = np.amin(depth_image)
	max_val = np.amax(depth_image) + 1
	# print(f'min, max = {min_val}, {max_val}')
	depth_image_vis = np.zeros(depth_image.shape, dtype=np.uint8)
	for y in range(depth_image.shape[0]):
		for x in range(depth_image.shape[1]):
			depth_image_vis[y, x] = int((depth_image[y, x] - min_val) / (max_val - depth_image[y, x])  * 255)

	return depth_image_vis

def main(outdir):
	# Makes sure outdir exists
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	# Setup pipeline to realsense
	pipeline = rs.pipeline()

	#Create a config and configure the pipeline to stream
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
	
	# Start streaming
	profile = pipeline.start(config)

	# Getting the depth sensor's depth scale (see rs-align example for explanation)
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	print("Depth Scale is: " , depth_scale)

	# We will be removing the background of objects more than clipping_distance_in_meters meters away
	clipping_distance_in_meters = 1 #1 meter
	clipping_distance = clipping_distance_in_meters / depth_scale

	# Create an align object
	# rs.align allows us to perform alignment of depth frames to others frames
	# The "align_to" is the stream type to which we plan to align depth frames.
	align_to = rs.stream.color
	align = rs.align(align_to)

	# Dumps first few due to exposure probs
	count = 0
	while count < 60:
		frames = pipeline.wait_for_frames()
		count += 1

	# Caps 30 rgbd frames
	count = 0
	while count < 30:
		# Align frames
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)

		# Get aligned frames
		aligned_depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()

		# Check we got em both
		if not aligned_depth_frame or not color_frame:
			continue

		# Get both images
		depth_image = np.asanyarray(aligned_depth_frame.get_data())
		# depth_image_vis = create_depth_vis(depth_image)
		depth_image_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		color_image = np.asanyarray(color_frame.get_data())
		# print(f'depth shape = {depth_image.shape}, color shape = {color_image.shape}')

		# Clip background set by clip distance
		# grey_color = 153
		# depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
		# g_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

		# Save images
		# fpath_depth = os.path.join(outdir, f'{str(count).zfill(2)}-depth.pgm')
		# # fpath_depth_vis = os.path.join(outdir, f'{str(count).zfill(2)}-depth_vis.pgm')
		# fpath_depth_vis = os.path.join(outdir, f'{str(count).zfill(2)}-depth_vis.ppm')
		# fpath_color = os.path.join(outdir, f'{str(count).zfill(2)}-color.ppm')

		fpath_depth = os.path.join(outdir, f'{str(count).zfill(2)}-depth.png')
		# fpath_depth_vis = os.path.join(outdir, f'{str(count).zfill(2)}-depth_vis.pgm')
		fpath_depth_vis = os.path.join(outdir, f'{str(count).zfill(2)}-depth_vis.png')
		fpath_color = os.path.join(outdir, f'{str(count).zfill(2)}-color.png')

		cv2.imwrite(fpath_depth, depth_image, [cv2.IMWRITE_PXM_BINARY, 0])
		cv2.imwrite(fpath_depth_vis, depth_image_vis, [cv2.IMWRITE_PXM_BINARY, 0])
		cv2.imwrite(fpath_color, color_image, [cv2.IMWRITE_PXM_BINARY, 0])
		count += 1

if __name__ == '__main__':
	outdir = sys.argv[1]
	main(outdir)