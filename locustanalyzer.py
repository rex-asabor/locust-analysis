import cv2
from matplotlib import pyplot as plt
import random, os
import numpy as np
from PIL import Image
import seaborn as sns
import math
from collections import deque
from enum import Enum
import copy

class LocustType(str, Enum):
    # May deprecate
    COHORT = "C"
    SINGLE = "S"

class LocustCorpus:
    def __init__(self, contours, dim):
        self.locusts = {} # maps locust label to locustEntity
        self.dim = dim
        self.num_cohorts = 0
        self.num_singles = 0
        for contour in contours:
            if len(contour) < 5: continue
            self.addLocustCntr(contour, LocustType.SINGLE)
        self.itr = 0
        
    def validateLocust(self, locust, minArea = 10):
        """ Returns true if contour meets criteria to become a locust entity """
        if locust.descriptors.area < minArea: return False
        return True

    def addLocustCntr(self, contour, locustType):   
        label = ""
        if locustType == LocustType.SINGLE:
            label = f"{locustType}:{self.num_singles}"
            self.num_singles += 1
        elif locustType == LocustType.COHORT:
            label = f"{locustType}:{self.num_cohorts}"
            self.num_cohorts += 1
        else:
            raise Exception("Unrecognized Locust Type")
            
        locust = LocustEntity(contour, locustType, label)
        if self.validateLocust(locust):
            self.locusts[label] = locust
        elif locustType == LocustType.SINGLE:
                self.num_singles -= 1
        elif locustType == LocustType.COHORT:
                self.num_cohorts -= 1
        
    @staticmethod
    def dist(l1, l2):
        x1,y1 = l1.center
        x2,y2 = l2.center
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def match(self, list1, list2, thresh=300):
        pairs = []    
        for l1 in list1:
            for l2 in list2:
                pairs.append((l1,l2, LocustCorpus.dist(l1, l2)))

        pairs.sort(key=lambda p: p[-1])
        remaining_l1 = set([l1.label for l1 in list1])
        remaining_l2 = set([l2.label for l2 in list2])
        matches = {}
        for l1, l2, dist_trav in pairs:
            if l1.label not in remaining_l1:
                continue
            elif l2.label not in remaining_l2:
                continue
            if not (l1.descriptors.area*.5 < l2.descriptors.area < l1.descriptors.area * 2) or dist_trav > thresh:
                continue

            px,py = l2.center
            w, h = self.dim
            if dist_trav > px or dist_trav > py or dist_trav > abs(w-px) or dist_trav > abs(h-py):
                # Identifying new locusts
                remaining_l2.remove(l2.label)
                edgeLocust = copy.copy(l2)
                edgeLocust.label = f"{LocustType.SINGLE}:{self.num_singles}"
                matches[edgeLocust.label] = edgeLocust
                self.num_singles += 1
                continue

            l1.update(l2.center, l2.descriptors, l2.contour)
            matches[l1.label] = l1

            remaining_l1.remove(l1.label)
            remaining_l2.remove(l2.label)

        for l2 in list2:
            if l2.label in remaining_l2:
                newLocust = copy.copy(l2)
                newLocust.label = f"{LocustType.SINGLE}:{self.num_singles}"
                matches[newLocust.label] = newLocust
                self.num_singles += 1

        return matches # l1 --> l2
    
    
    def update(self, other):
        """ Passes information from 'other' into 'self' corpus by mapping locusts"""
        num_visible_entities = len(self.locusts)
        seen_locusts = self.num_singles
        if seen_locusts < 1:   return other
       
        num_cohorts = self.num_cohorts
                
        matches = self.match(self.locusts.values(), other.locusts.values())        
        self.locusts = matches
        self.num_cohorts = num_cohorts
        self.itr += 1
        for locust in self.locusts.values():
            locust.descriptors.x_dir = self.get_direction(locust)
        return self
    
    
    def separateLocusts(self, other, selfLabel, otherLabels, num_singles):
        candidates = self.locusts[selfLabel].children
        # candidates should include *all descendants* 
        #     
        resultLocusts = []
        for otherLabel in otherLabels:
            otherLocust = other.locusts[otherLabel]
            if len(candidates) > 0:
                i, mostSimilar = min(enumerate(candidates), key=lambda l: self.dissimilarity(l[1], otherLocust))
                mostSimilar = copy.deepcopy(candidates.pop(i))
                mostSimilar.update(otherLocust.center, otherLocust.descriptors, otherLocust.contour)
                resultLocusts.append(mostSimilar)
                print(f"child separated from {selfLabel}on frame {self.itr}")
            else:
                otherLocust.label = f"{LocustType.SINGLE}:{num_singles}"
                num_singles += 1
                resultLocusts.append(otherLocust)   
        return resultLocusts, num_singles

        
    def nearestLocust(self, locust):
        """Returns label, LocustEntity """
        if self.cnt == 0:
            raise Exception("Locust Corpus is empty.")
            
        nearest = 0, self.locusts[0]
        for cand in self.locusts.items():
            nearest = min(cand, nearest, key = lambda pair: LocustCorpus.dist(pair[1], locust))
        return nearest
    
    def nearestKLocusts(self, locust, k = 5):
        nearestLocusts = sorted(self.locusts.values(), key = lambda l2: LocustCorpus.dist(locust, l2))
        return nearestLocusts[1:k+1]
        
    
    def get_direction(self, locust, thresh=10): # false = left, true = right
        if self.itr == 0:
            return False
        
        if len(locust.priors) > 0:
            # use data
            descriptor = locust.priors[-1]
            (px, py), _, _ = descriptor.ellipse
            x,y = locust.center
            if abs(x - px) < thresh:
                return descriptor.x_dir
            
            return px < x #locust travelled from left to right
        else:
            leftCnt = 0
            rightCnt = 0
            for neighbor in self.nearestKLocusts(locust):
                if len(locust.priors) == 0:
                    continue 
                if self.get_direction(neighbor):
                    rightCnt += 1
                else:
                    leftCnt += 1
            return rightCnt > leftCnt             
            
            
    def draw(self, image):
        for locust in self.locusts.values():
            x_dir = self.get_direction(locust)
            image = locust.draw(image)

        image = cv2.putText(
          img = image,
          text = f"Frame {self.itr}",
          org = (500,100),
          fontFace = cv2.FONT_HERSHEY_DUPLEX,
          fontScale = 1,
          color = (255, 0, 0),
          thickness = 3
        ) 
            
        return image
    
    def filter(self):
        # remove locusts that are outliers
        axes = [0 for _ in self.locusts.values()]
        for i, locust in enumerate(self.locusts.values()):
            ellipse = _,(d1,d2),_ = locust.descriptors.ellipse
            axes[i] = max(d1,d2)    
        q1 = np.quantile(axes, 0.25)
        q3 = np.quantile(axes, 0.75)
        k = 3
        outliers = set()
        for label, locust in self.locusts.items():
            ellipse = _,(d1,d2),_ = locust.descriptors.ellipse
            locust_axis = max(d1,d2) 
            if not q1 - k*(q3-q1) <= locust_axis <= q3 + k*(q3-q1):
                outliers.add(label)
        for outlier in outliers:
            del self.locusts[outlier]
                

class LocustEntity:
    class Descriptors:
        def __init__(self, contour):
            self.ellipse = center,(d1,d2),angle = cv2.fitEllipse(contour)
            area = cv2.contourArea(contour)
            self.axes = (d1,d2)
            self.angle = angle
            self.area = area
            self.x_dir = None
            
            
    def __init__(self, contour, locustType, label, prior_k=5):
        if not label.startswith(locustType.value): 
            raise Exception("Inconsistent type/ label")

        self.contour = contour
        self.type = locustType
        self.label = label
        self.descriptors = LocustEntity.Descriptors(contour)
        self.prior_k = prior_k

        self.center, _, _ = self.descriptors.ellipse 
    
        self.children = []
        self.priors = deque([])
        
    
    def getArea(self):
        return self.descriptors.area

                
    def update(self, center, descriptors, contour):
        self.center = center
        self.contour = contour
        self.priors.appendleft(self.descriptors)
        self.descriptors = descriptors
        while len(self.priors) > self.prior_k:
            self.priors.pop()
        
    def draw(self, image):   
        ellipse = (xc,yc),(d1,d2),angle = self.descriptors.ellipse
        rmajor = max(d1,d2)/2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        xtop = int(xc + math.cos(math.radians(angle))*rmajor)
        ytop = int(yc + math.sin(math.radians(angle))*rmajor)
        xbot = int(xc + math.cos(math.radians(angle+180))*rmajor)
        ybot = int(yc + math.sin(math.radians(angle+180))*rmajor)
        
        
        if self.descriptors.x_dir: # points right
            start_point = min((xtop, ytop), (xbot, ybot))
            end_point = max((xtop, ytop), (xbot, ybot))
        else:
            start_point = max((xtop, ytop), (xbot, ybot))
            end_point = min((xtop, ytop), (xbot, ybot))           
        
        color = (200, 200, 200)
        thickness = 2
        image = cv2.arrowedLine(image, start_point, end_point,
                    color, thickness, tipLength = 0.25)
    
        image = cv2.putText(
          img = image,
          text = str(self.label),
          org = (int(xc),int(yc)),
          fontFace = cv2.FONT_HERSHEY_DUPLEX,
          fontScale = 0.8,
          color = (0, 0, 255),
          thickness = 1
        )
        
        return image
    
class LocustAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.log = []
        
    def display(self):
        cap = cv2.VideoCapture(self.filename)
        mapping = {}
        prev_mapping = {}
        prev_centroids = []
        num_locusts = 0

        corpus = None
        visual = True
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):        
            ret, frame = cap.read()    
            if not ret: break
            open_cv_image = np.array(frame)
            scale_percent = 100 # percent of original size
            width = int(open_cv_image.shape[1] * scale_percent / 100)
            height = int(open_cv_image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image (for our visualization)
            resized = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_NEAREST )

            gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cal_range = (0,(np.median(gray_img)/1.8))
            bin_img = cv2.inRange(gray_img, *cal_range)


            ret,thresh = cv2.threshold(bin_img,0,100,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)    

            curr_corpus = LocustCorpus(contours, dim)
            curr_corpus.filter()
            if corpus is not None:
                curr_corpus = corpus.update(curr_corpus)

            if visual:
                image = curr_corpus.draw(resized)
                cv2.imshow("image", image)
                cv2.waitKey()
            else:
                #log/ write corpus data
                pass
            
            self.log.append(curr_corpus)
            corpus = curr_corpus
            
        cv2.destroyAllWindows() 