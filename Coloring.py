import cv2
import os
import numpy as np
import requests
import shutil
from enums.StagePaths import StagePaths
from werkzeug.contrib.cache import SimpleCache
from PIL import Image
from io import BytesIO
import skimage.segmentation
import scipy

cache = SimpleCache(threshold = 100, default_timeout = 0)



def lawnPercentage(geometryPoints, addressHash):
    sourcePath = 'data/address' + addressHash

    if not os.path.exists(sourcePath):
        os.makedirs(sourcePath)
    if (
        not os.path.exists(sourcePath + '/' + StagePaths.BOUNDLESS.value)
        or not os.path.exists(sourcePath + '/' + StagePaths.FILLED.value)
        or not os.path.exists(sourcePath + '/' + StagePaths.GREEN_MAP.value)
        or not os.path.exists(sourcePath + '/' + StagePaths.PLOT_MAP.value)
    ):
        saveAnalysisImages(geometryPoints,sourcePath)

    if not cache.has(addressHash) :
        return scanAnalysisImages(sourcePath, addressHash)
    else :
        return cache.get(addressHash)


def saveAnalysisImages(geometry, sourcePath):
    base = "https://maps.googleapis.com/maps/api/staticmap?size=800x800&zoom=18&maptype=satellite&path=weight%3A0"
    boundedBase = "%7Ccolor%3Ared%7Cfillcolor%3A0xff0000ff"
    key = "&key=AIzaSyCW2Iqv_sfKXruvHYl8b9YkUh7SbHUb9ZM"
    coordinates = makeCoordinates(geometry)
    rawMapURL = base + coordinates + key
    boundedMapURL = base + boundedBase + coordinates + key

    rawMap = requests.get(rawMapURL)
    boundedMap = requests.get(boundedMapURL)
    rawImage = Image.open(BytesIO(rawMap.content))
    boundedImage = Image.open(BytesIO(boundedMap.content))
    rawImage.save(sourcePath + '/' + StagePaths.BOUNDLESS.value)
    boundedImage.save(sourcePath + '/' + StagePaths.FILLED.value)

def makeCoordinates(geometryPoints):
    geometry = []
    for geoPoint in geometryPoints:
        geometry.append(str(geoPoint[1]))
        geometry.append(str(geoPoint[0]))
    name = ""
    swap = True
    for x in geometry:
        if(swap):
            name = name + "%7C"+ x
            swap = False
        else:
            name = name + "%2C" + x
            swap = True
    return name

def compareSegments(dist, pk, qk):
    pSeg = (np.array(dist[pk])+1)
    qSeg = (np.array(dist[qk])+1)
    if len(pSeg)>len(qSeg):
        pSeg = pSeg[np.random.choice(pSeg.shape[0],qSeg.shape[0], replace=False), :]
    elif len(qSeg)>len(pSeg):
        qSeg = qSeg[np.random.choice(qSeg.shape[0], pSeg.shape[0], replace=False), :]
    blueKL = np.sum(pSeg[:,0] * np.log(pSeg[:,0]/qSeg[:,0]))
    greenKL = np.sum(pSeg[:,1] * np.log(pSeg[:,1]/qSeg[:,1]))
    redKL = np.sum(pSeg[:,2] * np.log(pSeg[:,2]/qSeg[:,2]))

    return np.array([blueKL,greenKL,redKL])

# Get bounds of cropped image and generate a new, zoomed in image
# Output for non rectangular mask will have some empty space to fill in rectangular bounds
def resizeMaskedImage(img, redMask):
    minX = 2000
    maxX = 0
    minY = 2000
    maxY = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (redMask[i][j][2] >= 254 and redMask[i][j][0] == 0 and redMask[i][j][1] == 0):
                if (i > maxY):
                    maxY = i
                if (j > maxX):
                    maxX = j
                if (i < minY):
                    minY = i
                if (j < minX):
                    minX = j
    scaledDimensions = (600, int((maxY - minY) * (600.0 / (maxX - minX))))
    return cv2.resize(img[minY:maxY, minX:maxX], scaledDimensions, interpolation=cv2.INTER_AREA)

#Takes in target image predicts pixel coordinates of 'known' lawn
#outputs a green extracted lawn image
def greenPredict(img):
    targetOutput = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = img[i][j]
            blueGreenDiff = 0 + bgr[1] - bgr[0]
            redGreenDiff = 0 + bgr[1] - bgr[2]

            if ((redGreenDiff > 3) and (blueGreenDiff > 10) and (bgr[1] > 15)):
                targetOutput[i][j] = [0, 50, 0]
    kernel = np.ones((9, 9), np.uint8)
    img_erosion = cv2.erode(targetOutput, kernel, iterations=1)
    #erosionCoordinates = np.array(np.where(img_erosion[:, :, 1] == 50))
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    return img_erosion
    #greenCoordinates = np.array(np.where(img_dilation[:, :, 1] == 50))


def generateSegmentDistributions(img, segments):
    distributions = {}
    #lawnDistribution = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (segments[i][j] not in distributions.keys()):
                distributions[segments[i][j]] = [img[i][j]]
            else:
                distributions[segments[i][j]].append(img[i][j])
    return distributions


def greenSegmentPropagation(imgSegments, greenPrediction, distributions):
    greenCoordinates = np.array(np.where(greenPrediction[:, :, 1] == 50))
    greenSegmentIndices = np.unique(imgSegments[greenCoordinates[0, :], greenCoordinates[1, :]])
    allLawnSegments = []

    for seg in np.unique(imgSegments):
        if seg in greenSegmentIndices:
            allLawnSegments.append(seg)
        elif (compareSegments(distributions, seg, np.random.choice(greenSegmentIndices, 1)[0]).sum() > 7500
                and compareSegments(distributions, seg, np.random.choice(greenSegmentIndices, 1)[0]).sum() > 7500
                and compareSegments(distributions, seg, np.random.choice(greenSegmentIndices, 1)[0]).sum() > 7500
                and compareSegments(distributions, seg, np.random.choice(greenSegmentIndices, 1)[0]).sum() > 7500
                and compareSegments(distributions, seg, np.random.choice(greenSegmentIndices, 1)[0]).sum() > 7500):
            allLawnSegments.append(seg)

    output = np.zeros((imgSegments.shape[0], imgSegments.shape[1], 3))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if imgSegments[i][j] in allLawnSegments:
                output[i][j] = [0, 50, 0]
            else:
                output[i][j] = [0, 0, 0]
    return output

def calculateLawnPercentage(originalImage, lawnImage):
    fullCount = 0
    lawnCount = 0
    for i in range(originalImage.shape[0]):
        for j in range(originalImage.shape[1]):
            if (originalImage[i][j][0] > 0 or originalImage[i][j][1] > 0 or originalImage[i][j][2] > 0):
                fullCount += 1
            if lawnImage[i][j][1] > 0:
                lawnCount += 1
    return float(lawnCount / float(fullCount))

def greenSeek(sourcePath, fullImage, redMask):
    fullImage[np.where((redMask.all(axis=2) != [255, 0, 0]))] = [0, 0, 0]

    img = resizeMaskedImage(fullImage, redMask)
    initialPrediction = greenPredict(img)
    imgSegments = skimage.segmentation.slic(img, n_segments = 3000, sigma = 0)
    segmentDistributions = generateSegmentDistributions(img, imgSegments)
    finalPrediction = greenSegmentPropagation(imgSegments, initialPrediction, segmentDistributions)

    # images saved
    if not os.path.exists(sourcePath + '/' + StagePaths.PLOT_MAP.value):
        cv2.imwrite(os.path.join(sourcePath, StagePaths.PLOT_MAP.value), img)
    if not os.path.exists(sourcePath + '/' + StagePaths.GREEN_MAP.value):
        cv2.imwrite(os.path.join(sourcePath, StagePaths.GREEN_MAP.value), finalPrediction)

    return calculateLawnPercentage(img, finalPrediction)


def scanAnalysisImages(sourcePath, addressHash):
    raw = cv2.imread(sourcePath + '/' + StagePaths.BOUNDLESS.value)
    masked = cv2.imread(sourcePath + '/' + StagePaths.FILLED.value)




    percentage = greenSeek(sourcePath, raw, masked)
    cache.set(addressHash, percentage)
    return percentage


def getPercent(addressHash):
    return str(cache.get(addressHash))


def clearCache():
    cache.clear()
    shutil.rmtree("data")
    os.makedirs("data")
    return "True"