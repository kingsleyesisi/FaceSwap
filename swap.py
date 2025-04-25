import cv2
import sys

def detect_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    if len(faces) == 0:
        raise RuntimeError("No face detected.")
    # Just take the largest detected face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    print(x, y, w, h)
    return x, y, w, h

def swap_face(src_path, dst_path, out_path):
    # Load images
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    if src is None or dst is None:
        raise RuntimeError("Error loading images.")

    # Initialize the Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    sx, sy, sw, sh = detect_face(src, face_cascade)
    dx, dy, dw, dh = detect_face(dst, face_cascade)

    # Extract source face ROI and create mask
    face_src = src[sy:sy+sh, sx:sx+sw]
    mask = 254 * np.ones(face_src.shape[:2], face_src.dtype)
    print(mask)

    # Resize the source face to fit the destination face size
    face_src_resized = cv2.resize(face_src, (dw, dh))
    mask_resized = cv2.resize(mask, (dw, dh))

    # Coordinates for cloning center
    center = (dx + dw//2, dy + dh//2)

    # Seamlessly clone src face into dst image
    output = cv2.seamlessClone(
        face_src_resized,   # Source face image
        dst,                # Destination image
        mask_resized,       # Mask indicating where to clone
        center,             # Center of the region in dst
        cv2.NORMAL_CLONE    # Cloning method
    )

    # Save result
    cv2.imwrite(out_path, output)
    print(f"Face swapped image saved to {out_path}")

if __name__ == "__main__":
    import numpy as np
    if len(sys.argv) != 4:
        print("Usage: python fa ce_swap.py <source.jpg> <destination.jpg> <output.jpg>")
        sys.exit(1)
    swap_face(sys.argv[1], sys.argv[2], sys.argv[3])
