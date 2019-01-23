import csv


def read_samples_from_file(driving_log_filepath, steering_correction=0.25):
    image_paths = []
    measurements = []
    with open(driving_log_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip headers
        for line in reader:
            # camera images
            center = line[0].split('/')[-1]
            left = line[1].split('/')[-1]
            right = line[2].split('/')[-1]

            # measurements
            steering = float(line[3])
            throttle = float(line[4])
            brake = float(line[5])
            speed = float(line[6])

            # skip if speed is 0 because it's not representative for driving behavior
            if abs(speed) <= 0:
                continue

            image_paths.extend([center, left, right])
            measurements.extend([(steering, throttle, brake, speed),
                                 (steering + steering_correction, throttle, brake, speed),
                                 (steering - steering_correction, throttle, brake, speed)])

    return image_paths, measurements
