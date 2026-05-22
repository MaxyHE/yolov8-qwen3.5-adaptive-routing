import xml.etree.ElementTree as ET
import os

classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
           'chair','cow','diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor']

def convert_box(size, box):
    dw, dh = 1./size[0], 1./size[1]
    x = (box[0]+box[1])/2 * dw
    y = (box[2]+box[3])/2 * dh
    w = (box[1]-box[0]) * dw
    h = (box[3]-box[2]) * dh
    return x, y, w, h

anno_path = 'VOCdevkit/VOC2012/Annotations'
label_path = 'VOCdevkit/VOC2012/labels'
os.makedirs(label_path, exist_ok=True)

for xml_file in os.listdir(anno_path):
    if not xml_file.endswith('.xml'):
        continue
    tree = ET.parse(os.path.join(anno_path, xml_file))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txt_file = xml_file.replace('.xml', '.txt')
    with open(os.path.join(label_path, txt_file), 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            bb = obj.find('bndbox')
            box = (float(bb.find('xmin').text), float(bb.find('xmax').text),
                   float(bb.find('ymin').text), float(bb.find('ymax').text))
            x, y, w_n, h_n = convert_box((w, h), box)
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w_n:.6f} {h_n:.6f}\n")

print(f"转换完成，共{len(os.listdir(label_path))}个标签文件")
