import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read image
def get_512_px_gray_img(filename='images/Jannik CV.jpg'):
    img = cv2.imread(filename)

    # resize to max 512px in one direction
    max_px = max(img.shape[0], img.shape[1])
    scale_factor = max_px/512
    height = int(img.shape[0]/scale_factor)
    width = int(img.shape[1]/scale_factor)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# find edges using canny
def auto_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges

"""
# show edges with canny
binary_edges = auto_canny(gray)
plt.imshow(binary_edges, cmap='gray')

# find edges using laplacian:
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = laplacian - laplacian.min()
laplacian = laplacian / laplacian.max() * 255
laplacian = laplacian.astype('int')
plt.imshow(laplacian, cmap='gray')
plt.imshow(auto_canny(laplacian), cmap='gray')



# Harris Corner detection:
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
"""


def find_optimal_mirror_axis(img, x0_range = range(235, 255), x1_range = range(270, 290)):
    # axis is given by x0 and x1 which are intersections from axis with lower image edge and upper image edge

    rmses = pd.DataFrame(index=x0_range, columns=x1_range)
    i = 1
    for x0 in rmses.index:
        for x1 in rmses.columns:
            print(f'{i}/{len(x0_range)*len(x1_range)}')
            i = i+1
            mirrored_img = mirror_img_on_axis(img, x0, x1)
            rmses.loc[x0, x1] = rmse(img, mirrored_img)

    min_rmse = rmses.min().min()
    df_with_min = rmses.loc[rmses.min(axis=1) == min_rmse, rmses.min(axis=0) == min_rmse]

    return [df_with_min.index[0], df_with_min.columns[0]]


def mirror_img_on_axis(img, x0, x1):
    """

    :param img: ndarray with grayscale values
    :param x0: intersection from axis for mirroring with upper image edge
    :param x1: intersection from axis fpr mirroring with lower image edge
    :return: mirrored image
    """

    # get axis in form a + l*b (a,b in R^2)
    a = np.array([0, x0])
    b = np.array([img.shape[1], x1-x0])
    # normalize b
    b = b / ((b**2).sum())**0.5

    # get normalized vector (orthogonal to b)
    normal_vector = np.array([b[1], -b[0]])

    mirrored_img = np.zeros(img.shape)
    mirrored_img[:] = np.nan

    # get pixel values in mirrored matrix
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            p = np.array([i, j])
            # shift p
            p = p-a
            # get dist of p to axis:
            dist = (p[0]*b[1]-p[1]*b[0])

            # get mirrored point of p:
            mirrored_p = p + a - 2*dist*normal_vector
            mirrored_p = mirrored_p.astype(int)

            if (mirrored_p[0] >= 0) & (mirrored_p[0] < img.shape[0]) & \
                    (mirrored_p[1] >= 0) & (mirrored_p[1] < img.shape[1]):
                mirrored_img[i, j] = img[int(mirrored_p[0]), int(mirrored_p[1])]

    return mirrored_img


def rmse(img1, img2):
    error = img1 - img2
    return np.nanmean(error**2)**0.5


def show_img_with_axis(img, x0, x1):
    # insert black line in img:
    for i in range(0, img.shape[0]):
        j = int(x0 + ((x1-x0) * i / img.shape[0]))
        img[i, j] = 0
    plt.imshow(img, cmap='gray')


def show_both_imgs_with_axis(ax, img, x0, x1):
    mirrored_img = mirror_img_on_axis(img, x0, x1)
    average_image = np.array([img, mirrored_img]).mean(axis=0)
    # insert black line in img:
    for i in range(0, average_image.shape[0]):
        j = int(x0 + ((x1 - x0) * i / average_image.shape[0]))
        average_image[i, j] = 0
    ax.imshow(average_image, cmap='gray')


def get_mirrored_point(p, x0, x1):
    # get axis in form a + l*b (a,b in R^2)
    a = np.array([x0, 0])
    b = np.array([x1 - x0, img.shape[1]])
    # normalize b
    b = b / ((b ** 2).sum()) ** 0.5

    # get normalized vector (orthogonal to b)
    normal_vector = np.array([b[1], -b[0]])

    # shift p
    p = p - a
    # get dist of p to axis:
    dist = (p[0] * b[1] - p[1] * b[0])

    # get mirrored point of p:
    mirrored_p = p + a - 2 * dist * normal_vector
    mirrored_p = mirrored_p.astype(int)
    return mirrored_p


def get_nearest_points_for_each_point(all_points, no=3):
    if len(all_points) < no:
        raise ValueError(f'There are not enough points to calculate nearest {no} points')

    dists = np.zeros((len(all_points), len(all_points)))
    for idx_a, point_a in enumerate(all_points):
        for idx_b, point_b in enumerate(all_points):
            dists[idx_a, idx_b] = ((point_a-point_b)**2).sum()**0.5
    # set points with dist 0 to dist inf
    dists[dists == 0] = np.inf

    nearest_points = np.zeros((len(all_points), no))
    for idx, point in enumerate(all_points):
        for j in range(0, no):
            nearest_point = dists[idx, :].argmin()
            nearest_points[idx, j] = nearest_point
            dists[idx, nearest_point] = np.inf

    return nearest_points


def draw_lines_between_nearest_points(all_points, nearest_points):
    for idx, point in enumerate(all_points):
        for idx_of_near_points in nearest_points[idx, :]:
            draw_line_between_points(point, all_points[int(idx_of_near_points)])


def draw_line_between_points(point_a, point_b):
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], c='red')


def connect_nearest_points_without_intersections(all_points):
    points_to_connect = []

    dists = np.zeros((len(all_points), len(all_points)))
    for idx_a, point_a in enumerate(all_points):
        for idx_b, point_b in enumerate(all_points):
            dists[idx_a, idx_b] = ((point_a - point_b) ** 2).sum() ** 0.5
    # set values on and below diagonal to inf -> every dist is considered only once and not dist between point to itself
    # is considered:
    for n in range(0, dists.shape[0]):
        for m in range(0, n+1):
            dists[n, m] = np.inf

    while dists.min() < np.inf:
        idx_of_min = np.unravel_index(dists.argmin(), dists.shape)
        points_to_connect.append(idx_of_min)

        # set dist of found points to inf
        dists[idx_of_min] = np.inf
        point_a = all_points[idx_of_min[0]]
        point_b = all_points[idx_of_min[1]]
        # set dists of points separated by new found edge to inf
        for n in range(0, dists.shape[0]):
            for m in range(n + 1, dists.shape[0]):
                if dists[n, m] < np.inf:
                    if check_if_intersection(point_a, point_b, all_points[n], all_points[m]):
                        dists[n, m] = np.inf

    return points_to_connect


def check_if_intersection(point_a, point_b, point_c, point_d):
    """
    This function returns True if the connection of a and b and the connection of c and d intersect
    :return:
    """
    try:
        # get factors a and b of such that y = a + b*x describes line between point_a and point_b
        b = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        a = point_a[1] - b * point_a[0]
        # get factors c and d of such that y = c + d*x describes line between point_c and point_d
        d = (point_d[1] - point_c[1]) / (point_d[0] - point_c[0])
        c = point_c[1] - d * point_c[0]

        # calulate intersection by setting a + b*x = c + d*x
        x = (a - c) / (d - b)

        # check if x is between x coordinates of points a and b and between x coordinates of points c and d
        return (min(point_a[0], point_b[0]) < x) & (max(point_a[0], point_b[0]) > x) & \
               (min(point_c[0], point_d[0]) < x) & (max(point_c[0], point_d[0]) > x)

    except ZeroDivisionError:
        # this mean point a and b have same x coordinate or point c and d have same c coordinate -> deal with this
        if (point_b[0] == point_a[0]) & (point_c[0] == point_d[0]):  # both pairs have same x coordinate
            return point_a[0] == point_c[0]
        if point_b[0] == point_a[0]:
            # get factors c and d of such that y = c + d*x describes line between point_c and point_d
            d = (point_d[1] - point_c[1]) / (point_d[0] - point_c[0])
            c = point_c[1] - d * point_c[0]
            # get y value of connection between c and d at x = point_a[0]
            y = c + d*point_a[0]
            return (y > min(point_a[1], point_b[1])) & (y < max(point_a[1], point_b[1]))
        if point_c[0] == point_d[0]:
            # get factors a and b of such that y = a + b*x describes line between point_a and point_b
            b = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
            a = point_a[1] - b * point_a[0]
            # get y value of connection between a and b at x = point_c[0]
            y = a + b*point_c[0]
            return (y > min(point_c[1], point_d[1])) & (y < max(point_c[1], point_d[1]))


def plot_connections(all_points, points_to_connect):
    for points in points_to_connect:
        draw_line_between_points(all_points[points[0]], all_points[points[1]])


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(f'x={int(ix)}, y={int(iy)}')

    coords = np.array([ix, iy]).astype(int)
    mirrored_coords = get_mirrored_point(coords, x0, x1)

    ax.cla()
    show_both_imgs_with_axis(ax, img, x0, x1)

    ax.scatter(coords[0], coords[1], c='red')
    ax.scatter(mirrored_coords[0], mirrored_coords[1], c='red')
    plt.show()

    all_points.append(coords)
    all_points.append(mirrored_coords)

    # number of lines from each point
    no = 3
    if len(all_points) > no:
        # nearest_points = get_nearest_points_for_each_point(all_points=all_points, no=no)
        # draw_lines_between_nearest_points(all_points=all_points, nearest_points=nearest_points)
        points_to_connect = connect_nearest_points_without_intersections(all_points)
        plot_connections(all_points, points_to_connect)


global fig, ax, all_points, x0, x1, img

img = get_512_px_gray_img()

x0 = 242
x1 = 281

fig = plt.figure()
ax = fig.add_subplot(111)

show_both_imgs_with_axis(ax, img, x0, x1)

all_points = []


for i in range(0, 1):

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


