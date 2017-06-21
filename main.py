#!/usr/bin/python

import sys
import os
import PIL.Image
import pyocr
import pyocr.builders
import cv2
import numpy as np

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("       Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("       Available languages: %s" % ", ".join(langs))
lang = langs[1]
print("       Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.

print('       A demo script of the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')
if (len(sys.argv) < 2):
  print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
  quit()

pathname = os.path.dirname(sys.argv[0])

img      = cv2.imread(str(sys.argv[1]))
# for visualization
vis      = img.copy()

# Extract channels to be processed individually
channels = cv2.text.computeNMChannels(img)
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels)-1
for c in range(0,cn):
  channels.append((255-channels[c]))

# Apply the default cascade classifier to each independent channel (could be done in parallel)
print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
print("    (...) this may take a while (...)")
for channel in channels:

  erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
  er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

  erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
  er2 = cv2.text.createERFilterNM2(erc2,0.5)

  regions = cv2.text.detectRegions(channel,er1,er2)

  rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
  #rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

  #Visualization
  for r in range(0,np.shape(rects)[0]):
    rect = rects[r]
    cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
    cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)


#Visualization
cv2.imshow("Text detection result", vis)
new_img = PIL.Image.fromarray(vis[rects])




cv2.waitKey(0)
