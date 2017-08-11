import numpy as np

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def iou(box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def explit_c_mine(x):
    y = 1.0/(1.0 + np.exp(-x))
    return y

def box_constructor(meta, net_out_in):
    threshold = meta['thresh']
    classes = meta['labels']
    anchors = np.asarray(meta['anchors'])
    H, W, _ = meta['out_size']
    
    C = int(meta['classes'])
    B = int(meta['num'])
    net_out = net_out_in.reshape([H, W, B, int(net_out_in.shape[2]/B)])
    Classes = net_out[:,:,:,5:]
    Bbox_pred = net_out[:,:,:,:5]
    probs = np.zeros((H,W,B,C), dtype=np.float32)
    probs_filtered = np.zeros((H,W,B,C), dtype=np.float32)
    Bbox_pred[:,:,:,4] = explit_c_mine(Bbox_pred[:,:,:,4])
    offset = np.transpose(np.reshape(np.array([np.arange(19)]*95), (5,19,19)),(1,2,0))
    Bbox_pred[:,:,:,0] = (offset + explit_c_mine(Bbox_pred[:,:,:,0])) / W  
    Bbox_pred[:,:,:,1] = (np.transpose(offset, (1,0,2)) + explit_c_mine(Bbox_pred[:,:,:,1])) / H  
    for box_loop in range(B):
        Bbox_pred[:,:,box_loop,2] = np.exp(Bbox_pred[:,:,box_loop,2]) * anchors[2*box_loop + 0] /W
        Bbox_pred[:,:,box_loop,3] = np.exp(Bbox_pred[:,:,box_loop,3]) * anchors[2*box_loop + 1] /H
    
    
    class_probs = np.ascontiguousarray(Classes).reshape([H*W*B, C])
    max_all = np.max(class_probs, 1)
    max_all = np.expand_dims(max_all, 0)
    max_all = np.tile(max_all.T, (1, class_probs.shape[1]))

    class_probs = np.exp(class_probs - max_all)
    sum_all = np.sum(class_probs, 1)


    temp_pred = np.ascontiguousarray(Bbox_pred[:,:,:,4]).reshape([H*W*B, 1])
    temp_pred = np.tile(temp_pred, (1, class_probs.shape[1]))
    sum_all = np.expand_dims(sum_all, 0)
    sum_all = np.tile(sum_all.T, (1, class_probs.shape[1]))
    probs = class_probs * temp_pred /sum_all 
    probs = np.ascontiguousarray(probs).reshape([H, W, B, C])



    #apply score threshold
    bboxes = Bbox_pred[:,:,:,:4]
    filter_mat_probs = np.array(probs > threshold, dtype = 'bool')
    probs_filtered = probs[filter_mat_probs]
    filter_mat_bboxes = np.nonzero(filter_mat_probs)
    bboxes_filtered = bboxes[filter_mat_bboxes[0], filter_mat_bboxes[1], filter_mat_bboxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_bboxes[0], filter_mat_bboxes[1], filter_mat_bboxes[2]]
    
    #NMS
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    bboxes_filtered = bboxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]
    
    for i in range(len(probs_filtered)):
        if probs_filtered[i] == 0: continue
        for j in range(i+1, len(bboxes_filtered)):
            a = BoundBox(0)
            b = BoundBox(0) 
            a.x = bboxes_filtered[i, 0]
            a.y = bboxes_filtered[i, 1]
            a.w = bboxes_filtered[i, 2]
            a.h = bboxes_filtered[i, 3]
            b.x = bboxes_filtered[j, 0]
            b.y = bboxes_filtered[j, 1]
            b.w = bboxes_filtered[j, 2]
            b.h = bboxes_filtered[j, 3]
            
            if box_iou(a, b) > 0.4:
                probs_filtered[j] = 0
    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
    bboxes_filtered = bboxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]
    
    results = []
    numbox = len(bboxes_filtered)
    for i in range(len(bboxes_filtered)):
        result = dict()
        result['score'] = probs_filtered[i] 
        result['x1'] = bboxes_filtered[i][0] - bboxes_filtered[i][2]/2.0 
        result['y1'] = bboxes_filtered[i][1] - bboxes_filtered[i][3]/2.0  
        result['x2'] = bboxes_filtered[i][0] + bboxes_filtered[i][2]/2.0 
        result['y2'] = bboxes_filtered[i][1] + bboxes_filtered[i][3]/2.0
        result['x1'] = max(0.0, result['x1']) 
        result['y1'] = max(0.0, result['y1'])  
        result['x2'] = min(1.0, result['x2']) 
        result['y2'] = min(1.0, result['y2']) 
        result['label'] = classes_num_filtered[i]
        result['category'] = classes[classes_num_filtered[i]]
        results.append(result)
    #print(results)


    return results


