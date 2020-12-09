from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse.linalg
import cv2
from math import ceil, floor
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    plt.figure()
    plt.imshow(background_img)
    return object_img2, mask2

def upper_left_background_rc(object_mask, bottom_center):
    """ 
      Returns upper-left (row,col) coordinate in background image that corresponds to (0,0) in the object image
      object_mask: foreground mask in object image
      bottom_center: bottom-center (x=col, y=row) position of foreground object in background image
    """
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    upper_left_row = bottom_center[1]-int(max(ys)) 
    upper_left_col = bottom_center[0] - int(round(np.mean(xs)))
    return [upper_left_row, upper_left_col]

def crop_object_img(object_img, object_mask):
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    x1 = min(xs)-1
    x2 = max(xs)+1
    y1 = min(ys)-1
    y2 = max(ys)+1
    object_mask = object_mask[y1:y2, x1:x2]
    object_img = object_img[y1:y2, x1:x2, :]
    return object_img, object_mask

def get_combined_img(bg_img, object_img, object_mask, bg_ul):
    combined_img = bg_img.copy()
    (nr, nc) = object_img.shape[:2]

    for b in np.arange(object_img.shape[2]):
      combined_patch = combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b]
      combined_patch = combined_patch*(1-object_mask) + object_img[:,:,b]*object_mask
      combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b] =  combined_patch

    return combined_img 


def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    return mask


def num_equation(im, mask):
    im_h, im_w = im.shape 
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w) 
    
    nnz = (mask>0).sum()
    pointer = -np.ones(im.shape[0:2], dtype='int32')
    pointer[mask>0] = np.arange(nnz)
    
    e = 0;
    for y in range(im_h):
        for x in range(im_w): 
            if(x != im_w-1):
                e += 1   
            if(y != im_h-1):
                e += 1
    return e

def solve_constrain(im, mask, bg_im, bg_ul, neq):
    im_h, im_w = im.shape 
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w) 
    
    nnz = (mask>0).sum()
    pointer = -np.ones(im.shape[0:2], dtype='int32')
    pointer[mask>0] = np.arange(nnz)
    
    A = scipy.sparse.lil_matrix((neq, im_w * im_h), dtype='double')
    b = np.zeros((neq,1), dtype='double')
    e = 0;
    for y in range(im_h):
        for x in range(im_w): 
            bg_y = y + bg_ul[0]
            bg_x = x + bg_ul[1]
            if (x != im_w-1 and pointer[y][x+1] < 0):
                A[e, im2var[y][x]] = 1
                b[e] = (im[y][x] - im[y][x+1]) + bg_im[bg_y][bg_x+1]
                e += 1

            if (y != im_h-1 and pointer[y+1][x] < 0):
                A[e, im2var[y][x]] = 1
                b[e] = (im[y][x] - im[y+1][x]) + bg_im[bg_y+1][bg_x]
                e += 1

    for y in range(im_h):
        for x in range(im_w): 
            if (x != im_w-1 and pointer[y][x+1] >= 0):
                    A[e, im2var[y][x]] = 1
                    A[e, im2var[y][x+1]] = -1
                    b[e] = im[y][x] - im[y][x+1]
                    e += 1           

            if (y != im_h-1 and pointer[y+1][x] >= 0):
                    A[e, im2var[y][x]] = 1
                    A[e, im2var[y+1][x]] = -1
                    b[e] = im[y][x] - im[y+1][x]
                    e += 1
  
    print("finish preprocess")
    v = scipy.sparse.linalg.lsqr(A, b)
    
    result = np.array(v[0], dtype = float).reshape(im_h, im_w)
    
    # for i in range(im_h):
    #     for j in range(im_w):
    #         if (pointer[i][j] < 0):
    #             result[i][j] = bg_im[i+bg_ul[0]][j+bg_ul[1]]
    return result

def poisson_blend(object_img, object_mask, bg_img, bg_ul):
    """
    Returns a Poisson blended image with masked object_img over the bg_img at position specified by bg_ul.
    Can be implemented to operate on a single channel or multiple channels
    :param object_img: the image containing the foreground object
    :param object_mask: the mask of the foreground object in object_img
    :param background_img: the background image 
    :param bg_ul: position (row, col) in background image corresponding to (0,0) of object_img 
    """
    im = object_img
    neq = num_equation(im, object_mask)
    result = solve_constrain(im, object_mask, bg_img, bg_ul, neq)

    # s_max = np.max(result)
    # result /= s_max
    
    im_h, im_w = im.shape 
    bg = bg_img.copy()

    for i in range(im_h):
        for j in range(im_w):
            bg[i+bg_ul[0]][j+bg_ul[1]] = result[i][j]
    print("poisson done")
    return bg

def align_images(input_img_1, input_img_2, pts_img_1, pts_img_2,
                 save_images=False):
    
    # Load images
    im1 = input_img_1.copy()
    im2 = input_img_2.copy()
    plt.figure()
    plt.imshow(im1)

    # get image sizes
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    # Get center coordinate of the line segment
    center_im1 = np.mean(pts_img_1, axis=0)
    center_im2 = np.mean(pts_img_2, axis=0)
    

    # translate first so that center of ref points is center of image
    tx = np.around((w1 / 2 - center_im1[0]) * 2).astype(int)

    if tx > 0:
        im1 = np.r_['1', np.zeros((im1.shape[0], tx, 3)), im1]

    else:
        im1 = np.r_['1', im1, np.zeros((im1.shape[0], -tx, 3))]

    ty = np.round((h1 / 2 - center_im1[1]) * 2).astype(int)

    if ty > 0:
        im1 = np.r_['0', np.zeros((ty, im1.shape[1], 3)), im1]

    else:
        im1 = np.r_['0', im1, np.zeros((-ty, im1.shape[1], 3))]

    tx = np.around((w2 / 2 - center_im2[0]) * 2).astype(int)

    if tx > 0:
        im2 = np.r_['1', np.zeros((im2.shape[0], tx, 3)), im2]

    else:
        im2 = np.r_['1', im2, np.zeros((im2.shape[0], -tx, 3))]

    ty = np.round((h2 / 2 - center_im2[1]) * 2).astype(int)

    if ty > 0:
        im2 = np.r_['0', np.zeros((ty, im2.shape[1], 3)), im2]

    else:
        im2 = np.r_['0', im2, np.zeros((-ty, im2.shape[1], 3))]

    # downscale larger image so that lengths between ref points are the same
    len1 = np.linalg.norm(pts_img_1[0]-pts_img_1[1])
    len2 = np.linalg.norm(pts_img_2[0]-pts_img_2[1])
    dscale = len2 / len1

    if dscale < 1:
        width = int(im1.shape[1] * dscale)
        height = int(im1.shape[0] * dscale)
        dim = (width, height)
        im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_LINEAR)

    else:
        width = int(im2.shape[1] * 1 / dscale)
        height = int(im2.shape[0] * 1 / dscale)
        dim = (width, height)
        im2 = cv2.resize(im2, dim, interpolation=cv2.INTER_LINEAR)

    # rotate im1 so that angle between points is the same
    theta1 = np.arctan2(-(pts_img_1[:, 1][1]-pts_img_1[:, 1][0]),
                        pts_img_1[:, 0][1]-pts_img_1[:, 0][0])
    theta2 = np.arctan2(-(pts_img_2[:, 1][1]-pts_img_2[:, 1][0]),
                        pts_img_2[:, 0][1]-pts_img_2[:, 0][0])
    dtheta = theta2-theta1
    rows, cols = im1.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), dtheta*180/np.pi, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((rows * sin) + (cols * cos))
    nH = int((rows * cos) + (cols * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cols/2
    M[1, 2] += (nH / 2) - rows/2

    im1 = cv2.warpAffine(im1, M, (nW, nH))
    plt.imshow(im1)

    # Crop images (on both sides of border) to be same height and width
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    minw = min(w1, w2)
    brd = (max(w1, w2)-minw)/2
    if minw == w1:  # crop w2
        im2 = im2[:, ceil(brd):-floor(brd), :]
        tx = tx-ceil(brd)
    else:
        im1 = im1[:, ceil(brd):-floor(brd), :]
        tx = tx+ceil(brd)

    minh = min(h1, h2)
    brd = (max(h1, h2)-minh)/2
    if minh == h1:  # crop w2
        im2 = im2[ceil(brd):-floor(brd), :, :]
        ty = ty-ceil(brd)
    else:
        im1 = im1[ceil(brd):-floor(brd), :, :]
        ty = ty+ceil(brd)

    #im1 = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #im2 = cv2.cvtColor(im2.astype(np.uint8), cv2.COLOR_RGB2BGR)

    if save_images:
        output_img_1 = 'aligned_{}'.format(os.path.basename(input_img_1))
        output_img_2 = 'aligned_{}'.format(os.path.basename(input_img_2))
        cv2.imwrite(output_img_1, im1)
        cv2.imwrite(output_img_2, im2)

    return im1, im2
