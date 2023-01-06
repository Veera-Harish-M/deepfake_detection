import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
from blazeface import BlazeFace
from PIL import Image
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()

input_size = (128, 128)
front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")



def process_image( path: str = None, img: Image.Image or np.ndarray = None) -> dict:
    """
    Process a single image
    :param path: Path to the image
    :param img: image
    :return:
    """

    if img is not None and path is not None:
        raise ValueError('Only one argument between path and img can be specified')
    if img is None and path is None:
        raise ValueError('At least one argument between path and img must be specified')

    target_size = input_size

    if img is None:
        img = np.asarray(Image.open(str(path)))
    else:
        img = np.asarray(img)

    # Split the frames into several tiles. Resize the tiles to 128x128.
    tiles, resize_info = _tile_frames(np.expand_dims(img, 0), target_size)
    # tiles has shape (num_tiles, target_size, target_size, 3)
    # resize_info is a list of four elements [resize_factor_y, resize_factor_x, 0, 0]

    # Run the face detector. The result is a list of PyTorch tensors,
    # one for each tile in the batch.
    detections = front_net.predict_on_batch(tiles)

    # Convert the detections from 128x128 back to the original frame size.
    detections = _resize_detections(detections, target_size, resize_info)

    # Because we have several tiles for each frame, combine the predictions
    # from these tiles. The result is a list of PyTorch tensors, but now one
    # for each frame (rather than each tile).
    num_frames = 1
    frame_size = (img.shape[1], img.shape[0])
    detections = _untile_detections(num_frames, frame_size, detections)

    # Crop the faces out of the original frame.
    frameref_detections = _add_margin_to_detections(detections[0], frame_size, 0.2)
    faces = _crop_faces(img, frameref_detections)
    kpts = _crop_kpts(img, detections[0], 0.3)

    # Add additional information about the frame and detections.
    scores = list(detections[0][:, 16].cpu().numpy())
    frame_dict = {"frame_w": frame_size[0],
                    "frame_h": frame_size[1],
                    "faces": faces,
                    "kpts": kpts,
                    "detections": frameref_detections.cpu().numpy(),
                    "scores": scores,
                    }

    # Sort faces by descending confidence
    frame_dict = _soft_faces_by_descending_score(frame_dict)

    return frame_dict

def _tile_frames( frames: np.ndarray, target_size: Tuple[int, int]) -> (np.ndarray, List[float]):
    """Splits each frame into several smaller, partially overlapping tiles
    and resizes each tile to target_size.

    After a bunch of experimentation, I found that for a 1920x1080 video,
    BlazeFace works better on three 1080x1080 windows. These overlap by 420
    pixels. (Two windows also work but it's best to have a clean center crop
    in there as well.)

    I also tried 6 windows of size 720x720 (horizontally: 720|360, 360|720;
    vertically: 720|1200, 480|720|480, 1200|720) but that gives many false
    positives when a window has no face in it.

    For a video in portrait orientation (1080x1920), we only take a single
    crop of the top-most 1080 pixels. If we split up the video vertically,
    then we might get false positives again.

    (NOTE: Not all videos are necessarily 1080p but the code can handle this.)

    Arguments:
        frames: NumPy array of shape (num_frames, height, width, 3)
        target_size: (width, height)

    Returns:
        - a new (num_frames * N, target_size[1], target_size[0], 3) array
            where N is the number of tiles used.
        - a list [scale_w, scale_h, offset_x, offset_y] that describes how
            to map the resized and cropped tiles back to the original image
            coordinates. This is needed for scaling up the face detections
            from the smaller image to the original image, so we can take the
            face crops in the original coordinate space.
    """
    num_frames, H, W, _ = frames.shape

    num_h, num_v, split_size, x_step, y_step = get_tiles_params(H, W)

    splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

    i = 0
    for f in range(num_frames):
        y = 0
        for v in range(num_v):
            x = 0
            for h in range(num_h):
                crop = frames[f, y:y + split_size, x:x + split_size, :]
                splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                x += x_step
                i += 1
            y += y_step

    resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
    return splits, resize_info


def _resize_detections( detections, target_size, resize_info):
    """Converts a list of face detections back to the original
    coordinate system.

    Arguments:
        detections: a list containing PyTorch tensors of shape (num_faces, 17)
        target_size: (width, height)
        resize_info: [scale_w, scale_h, offset_x, offset_y]
    """
    projected = []
    target_w, target_h = target_size
    scale_w, scale_h, offset_x, offset_y = resize_info

    for i in range(len(detections)):
        detection = detections[i].clone()

        # ymin, xmin, ymax, xmax
        for k in range(2):
            detection[:, k * 2] = (detection[:, k * 2] * target_h - offset_y) * scale_h
            detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_w - offset_x) * scale_w

        # keypoints are x,y
        for k in range(2, 8):
            detection[:, k * 2] = (detection[:, k * 2] * target_w - offset_x) * scale_w
            detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_h - offset_y) * scale_h

        projected.append(detection)

    return projected


def _untile_detections( num_frames: int, frame_size: Tuple[int, int], detections: List[torch.Tensor]) -> List[
    torch.Tensor]:
    """With N tiles per frame, there also are N times as many detections.
    This function groups together the detections for a given frame; it is
    the complement to tile_frames().
    """
    combined_detections = []

    W, H = frame_size

    num_h, num_v, split_size, x_step, y_step = get_tiles_params(H, W)

    i = 0
    for f in range(num_frames):
        detections_for_frame = []
        y = 0
        for v in range(num_v):
            x = 0
            for h in range(num_h):
                # Adjust the coordinates based on the split positions.
                detection = detections[i].clone()
                if detection.shape[0] > 0:
                    for k in range(2):
                        detection[:, k * 2] += y
                        detection[:, k * 2 + 1] += x
                    for k in range(2, 8):
                        detection[:, k * 2] += x
                        detection[:, k * 2 + 1] += y

                detections_for_frame.append(detection)
                x += x_step
                i += 1
            y += y_step

        combined_detections.append(torch.cat(detections_for_frame))

    return combined_detections

def _add_margin_to_detections( detections: torch.Tensor, frame_size: Tuple[int, int],
                                margin: float = 0.2) -> torch.Tensor:
    """Expands the face bounding box.

    NOTE: The face detections often do not include the forehead, which
    is why we use twice the margin for ymin.

    Arguments:
        detections: a PyTorch tensor of shape (num_detections, 17)
        frame_size: maximum (width, height)
        margin: a percentage of the bounding box's height

    Returns a PyTorch tensor of shape (num_detections, 17).
    """
    offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
    detections = detections.clone()
    detections[:, 0] = torch.clamp(detections[:, 0] - offset * 2, min=0)  # ymin
    detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)  # xmin
    detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
    detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
    return detections

def _crop_faces( frame: np.ndarray, detections: torch.Tensor) -> List[np.ndarray]:
    """Copies the face region(s) from the given frame into a set
    of new NumPy arrays.

    Arguments:
        frame: a NumPy array of shape (H, W, 3)
        detections: a PyTorch tensor of shape (num_detections, 17)

    Returns a list of NumPy arrays, one for each face crop. If there
    are no faces detected for this frame, returns an empty list.
    """
    faces = []
    for i in range(len(detections)):
        ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int)
        face = frame[ymin:ymax, xmin:xmax, :]
        faces.append(face)
    return faces

def _crop_kpts( frame: np.ndarray, detections: torch.Tensor, face_fraction: float):
    """Copies the parts region(s) from the given frame into a set
    of new NumPy arrays.

    Arguments:
        frame: a NumPy array of shape (H, W, 3)
        detections: a PyTorch tensor of shape (num_detections, 17)
        face_fraction: float between 0 and 1 indicating how big are the parts to be extracted w.r.t the whole face

    Returns a list of NumPy arrays, one for each face crop. If there
    are no faces detected for this frame, returns an empty list.
    """
    faces = []
    for i in range(len(detections)):
        kpts = []
        size = int(face_fraction * min(detections[i, 2] - detections[i, 0], detections[i, 3] - detections[i, 1]))
        kpts_coords = detections[i, 4:16].cpu().numpy().astype(np.int)
        for kpidx in range(6):
            kpx, kpy = kpts_coords[kpidx * 2:kpidx * 2 + 2]
            kpt = frame[kpy - size // 2:kpy - size // 2 + size, kpx - size // 2:kpx - size // 2 + size, ]
            kpts.append(kpt)
        faces.append(kpts)
    return faces

def _soft_faces_by_descending_score( frame_dict: dict) -> dict:
    if len(frame_dict['scores']) > 1:
        sort_idxs = np.argsort(frame_dict['scores'])[::-1]
        new_faces = [frame_dict['faces'][i] for i in sort_idxs]
        new_kpts = [frame_dict['kpts'][i] for i in sort_idxs]
        new_detections = frame_dict['detections'][sort_idxs]
        new_scores = [frame_dict['scores'][i] for i in sort_idxs]
        frame_dict['faces'] = new_faces
        frame_dict['kpts'] = new_kpts
        frame_dict['detections'] = new_detections
        frame_dict['scores'] = new_scores
    return frame_dict
def get_tiles_params( H, W):
    split_size = min(H, W, 720)
    x_step = (W - split_size) // 2
    y_step = (H - split_size) // 2
    num_v = (H - split_size) // y_step + 1 if y_step > 0 else 1
    num_h = (W - split_size) // x_step + 1 if x_step > 0 else 1
    return num_h, num_v, split_size, x_step, y_step


# filenames = [ "1face.png", "0V5OIDW7FZ.jpg", "0UVZK7ZADZ.jpg" ]

# xfront = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8)
# xback = np.zeros((len(filenames), 256, 256, 3), dtype=np.uint8)

# for i, filename in enumerate(filenames):
#     img = cv2.imread(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     xback[i] = cv2.resize(img, (256, 256))



# back_detections= back_net.predict_on_batch(xback)
# i=0
# for d in back_detections:
#     plot_detections(xback[i], d)
#     i+=1
