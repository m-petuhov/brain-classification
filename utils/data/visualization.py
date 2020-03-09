import numpy as np
import cv2

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(image, plt, threshold=20):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, level=threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def show_slices(image, plt, plane='axial', rows=6, cols=6, figsize=(18, 18)):
    if plane not in ['axial', 'sagittal', 'coronal']:
        raise NameError(f"{plane} is not correct; correct modes: {['axial', 'sagittal', 'coronal']}")

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    n_slices = rows * cols
    depth = image.shape[2]
    step = depth // n_slices
    ind = 0

    for i in range(0, depth, step):
        if plane == 'sagittal':
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image[:, :, i], (224, 224)), 2), cmap='gray')
        elif plane == 'coronal':
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image[:, i, :], (224, 224)), 2), cmap='gray')
        else:
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image[i, :, :], (224, 224)), 2), cmap='gray')

        ax[min(ind // rows, rows - 1), ind % cols].set_title('slice %d' % i)
        ax[min(ind // rows, rows - 1), ind % cols].axis('off')
        ind += 1

    plt.show()


def show_slices_from2images(image1, image2, plt, plane='axial', rows=6, cols=6, figsize=(18, 18)):
    if plane not in ['axial', 'sagittal', 'coronal']:
        raise NameError(f"{plane} is not correct; correct modes: {['axial', 'sagittal', 'coronal']}")
    elif rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("Rows and cols should be even numbers")
    elif image1.shape != image2.shape:
        raise ValueError("Images shape should be equals")

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    n_slices = int(rows * cols / 2)
    depth = image1.shape[2]
    step = depth // n_slices
    ind = 0

    for i in range(0, depth, step):
        if plane == 'sagittal':
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image1[:, :, i], (224, 224)), 2), cmap='gray')
            ax[min(ind // rows, rows - 1), ind % cols + 1].imshow(np.rot90(cv2.resize(image2[:, :, i], (224, 224)), 2), cmap='gray')
        elif plane == 'coronal':
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image1[:, i, :], (224, 224)), 2), cmap='gray')
            ax[min(ind // rows, rows - 1), ind % cols + 1].imshow(np.rot90(cv2.resize(image2[:, i, :], (224, 224)), 2), cmap='gray')
        else:
            ax[min(ind // rows, rows - 1), ind % cols].imshow(np.rot90(cv2.resize(image1[i, :, :], (224, 224)), 2), cmap='gray')
            ax[min(ind // rows, rows - 1), ind % cols + 1].imshow(np.rot90(cv2.resize(image2[i, :, :], (224, 224)), 2), cmap='gray')

        ax[min(ind // rows, rows - 1), ind % cols].set_title('Image 1 slice %d' % i)
        ax[min(ind // rows, rows - 1), ind % cols].axis('off')
        ax[min(ind // rows, rows - 1), ind % cols + 1].set_title('Image 2 slice %d' % i)
        ax[min(ind // rows, rows - 1), ind % cols + 1].axis('off')
        ind += 2

    plt.show()
