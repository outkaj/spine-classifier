import os
import cv2
import scipy.misc as smp
import numpy as np
import json
import pprint
import sys
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import imutils

# Hardcoded pink color to highlight detected text region
color = (170, 28, 155)
char_height = 20.0
# color = (0, 0, 0)


def bbox(points):
    res = np.zeros((2, 2))
    res[0, :] = np.min(points, axis=0)
    res[1, :] = np.max(points, axis=0)
    return res


def bbox_width(bbox):
    return (bbox[1, 0] - bbox[0, 0] + 1)


def bbox_height(bbox):
    return (bbox[1, 1] - bbox[0, 1] + 1)


def aspect_ratio(region):
    bb = bbox(region)
    return (bbox_width(bb) / bbox_height(bb))


def filter_on_ar(regions):
    # Filter text regions based on Aspect-ration &amp;amp;amp;amp;lt;
    # 3.0&amp;quot;
    return [x for x in regions if aspect_ratio(x) < 3.0]


def dbg_draw_txt_contours(img, mser):
    # amp;quot;Draws contours on original image to show detected text
    # region&amp;quot;
    overlapped_img = cv2.drawContours(img, mser, -1, color)
    new_img = smp.toimage(overlapped_img)
    new_img.show()


def dbg_draw_txt_rect(img, bbox_list):
   img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR, dstCn=3)
   scratch_image_name = 'books.tmp.bmp'
   for b in bbox_list:
      pt1 = tuple(map(int, b[0]))
      pt2 = tuple(map(int, b[1]))
      img = cv2.rectangle(img, pt1, pt2, color, 1)
        # break
   new_img = smp.toimage(img)
   new_img.show()
   new_img.save(scratch_image_name)


def preprocess_img(img):
    #&amp;quot;Enhance contrast and resize the image&amp;quot;
    # create a CLAHE object (Arguments are optional).
    # It is adaptive localized hist-eq and also avoid noise
    # amplification with cliplimit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # Resize to match SVGA size
    height, width = img.shape
    # SVGA size is 800 X 600
    #if width > height:
        #scale = 800. / width
    #else:
        #scale = 600. / width
    # Avoid shrinking
    #if scale < 1.0:
        #scale = 1.0
    #dst = cv2.resize(img, (0,0), None, scale, scale, cv2.INTER_LINEAR)
    #return dst
    return img


def swt_window_func(l):
    center = l[4]
    filtered_l = np.append(l[:4], l[5:])
    res = [n for n in filtered_l if n < center]
    return res


def swt(gimg):
    # TODO: fix threshold logically
    threshold = 90
    maxval = 255
    # THRESH_BINARY_INV because we want to find distance from foreground pixel
    # to background pixel
    temp, bimg = cv2.threshold(gimg, threshold, maxval, cv2.THRESH_BINARY_INV)
    rows, cols = bimg.shape
    # Pad 0 pixel on bottom-row to avoid Infinite distance
    row_2_pad = np.zeros([1, cols], dtype=np.uint8)
    bimg_padded = np.concatenate((bimg, row_2_pad), axis=0)
    dist = cv2.distanceTransform(
    bimg_padded,
    cv2.DIST_L2,
     cv2.DIST_MASK_PRECISE)
    dist = np.take(dist, list(range(rows)), axis=0)
    dist = dist.round()
    # print dist
    it = np.nditer([bimg, dist],
                   op_flags=[['readonly'], ['readonly']],
                   flags=['multi_index', 'multi_index'])

    # Look-up operation
    # while not it.finished:
    lookup = []
    max_col = 0
    max_row = 0
    for cur_b, cur_d in it:
        if it.multi_index[0] > max_row:
            max_row = it.multi_index[0]
        if it.multi_index[1] > max_col:
            max_col = it.multi_index[1]
        if cur_b:
            cur_lup = []
            pval = cur_d
            row = it.multi_index[0]
            if row != 0:
                row_l = row - 1
            else:
                row_l = row
            if row != rows - 1:
                row_u = row + 1
            else:
                row_u = row
            row_list = [row_l, row, row_u]
            col = it.multi_index[1]
            if col != 0:
                col_l = col - 1
            else:
                col_l = col
            if col != cols - 1:
                col_u = col + 1
            else:
                col_u = col
            col_list = [col_l, col, col_u]
            # TODO: avoid for loop for look-up operation
            for i in row_list:
                for j in col_list:
                    if i != row and j != col:
                        cur = dist[i, j]
                        if cur < pval:
                            cur_lup.append((i, j))
            lookup.append(cur_lup)
        else:
            lookup.append(None)
        # it.iternext()
    lookup = np.array(lookup)
    lookup = lookup.reshape(rows, cols)
    d_max = int(dist.max())
    for stroke in np.arange(d_max, 0, -1):
        stroke_index = np.where(dist == stroke)
        stroke_index = [(a, b)
                         for a, b in zip(stroke_index[0], stroke_index[1])]
        for stidx in stroke_index:
            neigh_index = lookup[stidx]
            for nidx in neigh_index:
                dist[nidx] = stroke

    it.reset()
    sw = []
    for cur_b, cur_d in it:
        if cur_b:
            sw.append(cur_d)
    return sw


def get_swt_frm_mser(region, rows, cols, img):
    #&amp;quot;Given image and total rows and columns, extracts SWT values from MSER region&amp;quot;
    bb = bbox(region)
    xmin = int(bb[0][0])
    ymin = int(bb[0][1])
    width = int(bbox_width(bb))
    height = int(bbox_height(bb))
    selected_pix = []
    xmax = xmin + width
    ymax = ymin + height
    for h in range(ymin, ymax):
        row = np.take(img, (h, ), axis=0)
        horz_pix = np.take(row, list(range(xmin, xmax)))
        selected_pix.append(horz_pix)
    selected_pix = np.array(selected_pix)
    sw = swt(selected_pix)
    return sw


def filter_on_sw(region_dict):
    filtered_dict = {}
    distance_th = 4.0
    group_num = 0
    for rkey in list(region_dict.keys()):
        med = region_dict[rkey]['sw_med']
        height = bbox_height(region_dict[rkey]['bbox'])
        added = False
        for fkey in filtered_dict:
            for k in filtered_dict[fkey]:
                elem_med = filtered_dict[fkey][k]['sw_med']
                elem_height = bbox_height(filtered_dict[fkey][k]['bbox'])
                m_ratio = med / elem_med
                h_ratio = height / elem_height
                if m_ratio > 0.66 and m_ratio < 1.5 and h_ratio > 0.5 and h_ratio < 2.0:
                    filtered_dict[fkey][rkey] = region_dict[rkey]
                    added = True
                    break
            if added:
                break
        if not added:
            name = 'group' + str(group_num)
            filtered_dict[name] = {}
            filtered_dict[name][rkey] = region_dict[rkey]
            group_num = group_num + 1
    return filtered_dict


def get_y_center(bb):
        ll = bb[0]
        ur = bb[1]
        return ((ll[1] + ur[1]) / 2.0)


def kmean(region_dict, rows, num_clusters):
    print("K-mean START ...")
    clusters = (float(rows) / num_clusters) * np.arange(num_clusters)
    cluster_vld = [True] * num_clusters
    # calculate initial cost assuming all regions assigned to cluster-0
    cost = 0.0
    for rkey in region_dict:
        center_y = get_y_center(region_dict[rkey]['bbox'])
        cost += center_y * center_y
    cost = cost / len(list(region_dict.keys()))

    iter_no = 0
    while True:
        iter_no = iter_no + 1
        # Assign cluster-id to each region
        for rkey in region_dict:
            center_y = get_y_center(region_dict[rkey]['bbox'])
            dist_y = np.abs(clusters - center_y)
            cluster_id = dist_y.argmin()
            region_dict[rkey]['clid'] = cluster_id

        # find new cost with assigned clusters
        new_cost = 0.0
        for i, c in enumerate(clusters):
            if cluster_vld[i]:
                num_regions = 0
                cluster_cost = 0.0
                for rkey in region_dict:
                    if(region_dict[rkey]['clid'] == i):
                        center_y = get_y_center(region_dict[rkey]['bbox'])
                        cluster_cost += (center_y - clusters[i]) ** 2
                        num_regions += 1
                if num_regions:
                    cluster_cost /= num_regions
            new_cost += cluster_cost

        # Stop when new cost is within 5% of old cost
        if new_cost >= 0.95 * cost:
            break
        else:
            cost = new_cost

        for i, c in enumerate(clusters):
            if cluster_vld[i]:
                num_regions = 0
                clusters[i] = 0.0
                for rkey in region_dict:
                    if(region_dict[rkey]['clid'] == i):
                        center_y = get_y_center(region_dict[rkey]['bbox'])
                        clusters[i] += center_y
                        num_regions += 1
                if num_regions:
                    clusters[i] = clusters[i] / num_regions
                else:
                    cluster_vld[i] = False

    # Merge nearby clusters
    for i, cur_cl in enumerate(clusters):
        if cluster_vld[i]:
            for j, iter_cl in enumerate(clusters):
                if abs(cur_cl - iter_cl) <= (char_height / 2.0) and i != j:
                    cluster_vld[j] = False
                    for rkey in region_dict:
                        # Update cluster-id to updated one
                        if region_dict[rkey]['clid'] == j:
                            region_dict[rkey]['clid'] = i

    print("K-mean DONE...")
    return cluster_vld


def dbg_get_cluster_rect(cluster_vld, region_dict):
    bbox_list = []
    for cl_no, vld in enumerate(cluster_vld):
        if vld:
            cur_lL = [100000, 100000]
            cur_uR = [-100000, -100000]
            for rkey in region_dict:
                if region_dict[rkey]['clid'] == cl_no:
                    region_lL = region_dict[rkey]['bbox'][0]
                    region_uR = region_dict[rkey]['bbox'][1]
                    # update min/max of x/y
                    if region_lL[0] < cur_lL[0]:
                        cur_lL[0] = region_lL[0]
                    if region_lL[1] <= cur_lL[1]:
                        cur_lL[1] = region_lL[1]
                    if region_uR[0] >= cur_uR[0]:
                        cur_uR[0] = region_uR[0]
                    if region_uR[1] >= cur_uR[1]:
                        cur_uR[1] = region_uR[1]
            bbox_list.append([cur_lL, cur_uR])
    return bbox_list


def get_bbox_img(gimg, bb):
    # print bb, gimg.shape
    y_start = int(bb[0][1])
    y_end = int(bb[1][1])
    x_start = int(bb[0][0])
    x_end = int(bb[1][0])
    # print x_start, x_end, y_start, y_end
    row_extracted = gimg.take(list(range(y_start, y_end + 1)), axis=0)
    # print gimg
    extracted = row_extracted.take(list(range(x_start, x_end + 1)), axis=1)
    return extracted


def get_text_from_cluster(cluster_vld, region_dict, gimg):
    bbox_list = dbg_get_cluster_rect(cluster_vld, region_dict)
    # scratch_image_name = 'books.tmp.bmp'
    str_list = []
    for bb in bbox_list:
      extracted = get_bbox_img(gimg, bb)
      # print extracted
      ext_img = smp.toimage(extracted)
      found = pytesseract.image_to_string(ext_img)
      str_list.append(found.strip())
    str_list.insert(0, str_list)
    print("TEXT FOUND");
    pprint.pprint(str_list)
    # cv2.imwrite('testverticalchanged.jpg', ext_img)
    # print(pytesseract.image_to_string(Image.open('testverticalchanged.jpg'),
    # lang='eng'))


def run(fimage):
    # Constants:
    ar_thresh_max = 3.0
    ar_thresh_min = 0.5
    sw_ratio_thresh = 0.5
    org_img = cv2.imread(fimage)
    # loop over the angles to rotate the image
    #for angle in xrange(0, 360, 90):
	# rotate the image and display it
	   # rotated = imutils.rotate(org_img, angle=angle)
	#cv2.imshow("Angle=%d" % (angle), rotated)
    gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_img = preprocess_img(gray_img)
    mser = cv2.MSER_create()
    mser.setDelta(4)
    mser_areas, _ = mser.detectRegions(gray_img)
    region_dict = {}
    rows, cols = gray_img.shape
    print(("Image size = {} X {}  MSER_AREAS = {} path = {}".format(rows, cols, len(mser_areas), fimage)))
    region_num = 0
    for m in mser_areas:
        name = 'mser_' + str(region_num)
        bb = bbox(m)
        ar = bbox_width(bb)/bbox_height(bb)
        # Filter based on AspectRatio
        if ar < ar_thresh_max: # and ar&amp;amp;amp;amp;gt;ar_thresh_min: #commented min check because '1' is getting filtered
            # print &amp;quot;SW for region: &amp;quot;, region_num
            sw = get_swt_frm_mser(m, rows, cols, gray_img)
            sw_std = np.std(sw)
            sw_mean = np.mean(sw)
            sw_ratio = sw_std/sw_mean
            # 2nd filter based on Stroke-Width
            if sw_ratio < sw_ratio_thresh:
                sw_med = np.median(sw)
                region_dict[name] = {'bbox': bb, 'sw_med': sw_med};
                region_num = region_num + 1

    num_clusters = int(rows/char_height)
    cluster_vld = kmean(region_dict, rows, num_clusters)
    bbox_list = dbg_get_cluster_rect(cluster_vld, region_dict)
    get_text_from_cluster(cluster_vld, region_dict, gray_img)
    cpy_img = np.copy(gray_img)
    dbg_draw_txt_rect(cpy_img, bbox_list)

if __name__ == '__main__':
    # db_path = 'src/' ### Edit this line
    # img_name = r'testhorizontal2.jpg'
    # img_name = r'testvertical.jpg'
    # img_name = r'cropped.png'
    # img_name = r'cropped2.png'
    # img_name = r'real_img.png'
    # img_name = r'Real1.JPG'
    # fimage = os.path.join(img_name)
    run("testhorizontal.jpg")
