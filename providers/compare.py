import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class Compare:
    def __init__(self, img1):
        self.img1_color = img1
        self.img1 = cv2.cvtColor(self.img1_color, cv2.COLOR_BGR2GRAY)
        self.img2 = None
        self.path2 = None

    def set(self, path2):
        self.path2 = path2
        self.img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    def basic(self):
        # Load images in grayscale
        img1 = self.img1
        img2 = self.img2

        # Ensure both images are the same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute template matching
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)

        # Extract the best match score (higher is better)
        similarity_score = np.max(result)

        return similarity_score  # Returns a float between -1 and 1


    def ssim(self):
        img1 = self.img1
        img2 = self.img2

        # Resize to match shapes
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM (1 = identical, -1 = completely different)
        score, _ = ssim(img1, img2, full=True)
        return score


    def orb(self):
        img1 = self.img1
        img2 = self.img2

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect and compute ORB descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Use BFMatcher to find best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity score (lower distance = better match)
        score = sum(match.distance for match in matches) / len(matches)

        return score  # Lower is better (0 = identical images)

    def sift(self):
        img1 = self.img1
        img2 = self.img2

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect and compute SIFT descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Use FLANN matcher for SIFT (faster than brute-force)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Return match count as a similarity score
        return len(good_matches)  # Higher is better

    def histogram(self):
        # Don't use the grayscale images
        img1 = self.img1_color
        img2 = cv2.imread(self.path2)

        # Convert to HSV for better color matching
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        # Compute color histograms
        hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize and compare using correlation
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return similarity  # 1 = identical, 0 = no match
