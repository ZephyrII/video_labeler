import os
import cv2
import xml.etree.ElementTree as ET


class BoxSaver():
    def __init__(self, save_dir, video_file):
        self.video_file = video_file
        self.save_dir = save_dir
        self.save_as_video = False
        self.video_handle = None
        self.label_file = '%s/labels.txt' % (self.save_dir)
        directory = '%s/annotations' % (self.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
            directory = '%s/images' % (self.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save(self, im, frame_id, boxes_iter):
        if self.save_as_video:
            roi = list(boxes_iter)[0][0]
            if self.video_handle == None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
                try:
                    video_filename = self.video_file[:-4] + "_cropped" + self.video_file[-4:]
                except Exception as e:
                    print(e)
                print(video_filename)
                self.video_handle = cv2.VideoWriter(video_filename, fourcc, 20.0,
                                                    (abs(roi[0][0] - roi[1][0]), abs(roi[0][1] - roi[1][1])))

            try:
                cropped_im = im[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
                self.video_handle.write(cropped_im)
            except Exception as e:
                print(e)
        else:
            for idx, (box, label) in enumerate(boxes_iter):
                p0, p1 = box
                video_file = self.video_file.split('/')[-1][:-4]
                filename = "%s_%d_%d" % (video_file, frame_id, idx)
                with open("%s/annotations/%s.xml" % (self.save_dir, filename), 'w') as f:
                    # write format frame_id label x0 y0 x1 y1
                    f.write(self.makeXml(p0, p1, label, im.shape[1], im.shape[0], filename))
                filename = '%s/images/%s.jpg' % (self.save_dir, filename)
                dir = os.path.split(filename)[0]
                if not os.path.exists(dir):
                    os.makedirs(dir)
                filename = os.path.abspath(filename)
                # obj_im=im[p0[1]:p1[1],p0[0]:p1[0]]
                cv2.imwrite(filename, im)

    def makeXml(self, p0, p1, className, imgWidth, imgHeigth, filename):
        ann = ET.Element('annotation')
        ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename + ".jpg"
        ET.SubElement(ann, 'path')
        source = ET.SubElement(ann, 'source')
        ET.SubElement(source, 'database').text = "Unknown"
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeigth)
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'segmented').text = "0"
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = className
        ET.SubElement(object, 'pose').text = "Unspecified"
        ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(p0[0])
        ET.SubElement(bndbox, 'ymin').text = str(p0[1])
        ET.SubElement(bndbox, 'xmax').text = str(p1[0])
        ET.SubElement(bndbox, 'ymax').text = str(p1[1])
        return ET.tostring(ann, encoding='unicode', method='xml')

    def release_video_file(self):
        self.video_handle.release()
