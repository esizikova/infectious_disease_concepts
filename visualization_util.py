import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
from colorsys import hls_to_rgb
from tqdm import tqdm


# Heat map visual# Heat map visualization
def show_heatmaps1(imgs, masks, K, colors1, enhance=1, savename=''):
    f = plt.figure(figsize=(len(imgs), 10))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        img = imgs[i]
        if img.max()<=1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2],4))
            for c in range(3): 
                layer[:,:,c] = colors1[k][c] * layer[:,:,c]
            mask = masks[i][k]
            layer[:,:,3] = mask
            plt.imshow(layer)
            plt.axis('off')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    if savename !='':
        plt.savefig(savename,bbox_inches='tight')
    plt.show()

def get_distinct_colors(n):
    ''' https://stackoverflow.com/questions/37299142/how-to-set-a-colormap-which-can-give-me-over-20-distinct-colors-in-matplotlib'''
    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))
    return colors


def visualize(images, heatmaps, input_labels, MAX_SHOW,save_prefix):
    SAVE = 1
    color_map = get_distinct_colors(max(7,heatmaps.shape[1]))
    saved_label_image_names = [] 
    MAX_SHOW=5
    return_indices = []
    for label in np.unique(input_labels):
        return_indices_label = []
        print('label ', label)
        indices_class = np.where(input_labels==label)
        if len(indices_class[0])==1:
            indices_class = indices_class[0]
        values_all = []
        for heatmap_id in range(heatmaps.shape[1]):
            heatmap_sel = heatmaps[:,heatmap_id,:,:]
            heatmap_sel = np.expand_dims(heatmap_sel,axis=1)

            heatmaps_sel = heatmap_sel[indices_class]
            heatmaps_sel = heatmaps_sel.reshape(heatmaps_sel.shape[0], heatmaps_sel.shape[1],-1)
            value = np.mean(np.sum(heatmaps_sel,axis=2),axis=0)
            values_all.append(value[0])

        ind_sort = np.argsort(values_all)[::-1]
        return_indices_sort = []
        for heatmap_id in ind_sort:#[:3]:
            heatmap_sel = heatmaps[:,heatmap_id,:,:]
            heatmap_sel = np.expand_dims(heatmap_sel,axis=1)

            #print assignment value
            heatmaps_sel = heatmap_sel[indices_class]
            heatmaps_sel = heatmaps_sel.reshape(heatmaps_sel.shape[0], heatmaps_sel.shape[1],-1)
            # visualize
            heatmap_sel[heatmap_sel<0.1] = 0
            heatmap_sel[heatmap_sel>=0.1] = 0.5
            
            color_selected = color_map[heatmap_id]
            if SAVE:
                savename= str(label) + '_' + str(heatmap_id) + '.png'
                images_sel_draw = images[indices_class[0]]
                return_indices_sort.append(indices_class[0][0:MAX_SHOW])
                if len(images_sel_draw.shape) < 4:
                    images_sel_draw = images_sel_draw.unsqueeze(0) #np.expand_dims(images_sel_draw,0)
                show_heatmaps1(images_sel_draw[0:MAX_SHOW].cpu().permute(0,2,3,1), 
                                                    heatmap_sel[indices_class][0:MAX_SHOW], 
                                                    1, [color_selected], enhance=0.3, 
                                                    savename=savename)
            return_indices_label.append(return_indices_sort)
        if SAVE:
            images_all = [Image.open(x) for x in [str(label) + '_' + str(heatmap_id)+'.png' for heatmap_id in ind_sort]]
            widths, heights = zip(*(i.size for i in images_all))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255))

            x_offset = 0
            for im in images_all:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            new_im.save(save_prefix+str(label)+'.png')
            print('saved to ' + save_prefix+str(label)+'.png')
            saved_label_image_names.append(save_prefix+str(label)+'.png')
            to_remove = [os.remove(x) for x in [str(label) + '_' + str(heatmap_id)+'.png' for heatmap_id in ind_sort]]
        return_indices.append(return_indices_label)
    # merge vertically    
    images_all = [Image.open(x) for x in saved_label_image_names]
    widths, heights = zip(*(i.size for i in images_all))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255))

    y_offset = 0
    for im in images_all:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    new_im.save(save_prefix+'all1.png')
    print('saved to ' + save_prefix+'all1.png')
    to_remove = [os.remove(x) for x in saved_label_image_names]
    return return_indices

def compute_evaluation_metrics(gt_masks, heatmaps,input_labels):
    '''
    compute iou and segmentation purity
    '''
    for label in np.unique(input_labels): # go by class
        indices_class = np.where(input_labels==label)
        if len(indices_class[0])==1:
            indices_class = indices_class[0]

        l_mean_metric_iou_all = []
        l_max_metric_seg_purity_all = []
        for heatmap_id in list(range(heatmaps.shape[1])):
            heatmap_sel = heatmaps[:,heatmap_id,:,:]
            heatmap_sel = np.expand_dims(heatmap_sel,axis=1)

            #print assignment value
            heatmaps_sel = heatmap_sel[indices_class]
            heatmaps_sel = heatmaps_sel.reshape(heatmaps_sel.shape[0], heatmaps_sel.shape[1],-1)

            heatmap_sel[heatmap_sel<0.1] = 0
            heatmap_sel[heatmap_sel>=0.1] = 0.5

            images_sel_draw = gt_masks[indices_class[0]]
            if len(images_sel_draw.shape) < 4:
                images_sel_draw = np.expand_dims(images_sel_draw,0) #images_sel_draw.unsqueeze(0) #np.expand_dims(images_sel_draw,0)
            color_selected=[1,0,0]

            l_metric_iou = []
            l_metric_seg_purity = []
            for img_id in range(images_sel_draw.shape[0]): # iterate over all images
                # gt
                # img_sel = images_sel_draw.cpu().permute(0,2,3,1)[0][:,:,0].numpy()
                img_sel = images_sel_draw[0][:,:,0]
                img_sel =  1.0 - (img_sel>0.0).astype(float)
                gt_mask_float = img_sel
                gt_mask_bool = img_sel.astype('bool')

                # pred
                heatmap0_sel = heatmap_sel[indices_class][img_id,0]
                pred_mask_float = (heatmap0_sel>0.0).astype(float)
                pred_mask_bool = pred_mask_float.astype('bool')

                # calculate metrics
                overlap = gt_mask_bool*pred_mask_bool # Logical AND
                union = gt_mask_bool + pred_mask_bool # Logical OR
                if float(union.sum())>0.0:
                    metric_iou = overlap.sum()/float(union.sum()) 
                else:
                    metric_iou = 0.0
                if float(pred_mask_bool.sum())>0.0:
                    metric_seg_purity = overlap.sum()/float(pred_mask_bool.sum())
                else:
                    metric_seg_purity = 0.0   

                l_metric_iou.append(metric_iou)
                l_metric_seg_purity.append(metric_seg_purity)

                # visualize
                #print("iou ", metric_iou, " purity ", metric_seg_purity)
                #plt.imshow(gt_mask_float,cmap='gray')
                #plt.imshow(pred_mask_float,alpha=0.5)
                #plt.show()
                
            mean_metric_iou = np.mean(metric_iou)
            max_metric_seg_purity = np.max(metric_seg_purity)
            l_mean_metric_iou_all.append(mean_metric_iou)
            l_max_metric_seg_purity_all.append(max_metric_seg_purity)
            
    return [np.mean(l_mean_metric_iou_all),np.max(l_max_metric_seg_purity_all)]

    

def intersection_over_union(hmap1,hmap2):
    overlap = hmap1*hmap2 # Logical AND
    union = hmap1 + hmap2 # Logical OR
    metric_iou = 0
    if float(union.sum())>0.0:
        metric_iou = overlap.sum()/float(union.sum()) 
    return metric_iou

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calculate_heatmap_matches(heatmaps1, heatmaps2, THRESHOLD=0.3):
    arr_ind_match = np.zeros((heatmaps1.shape[0],heatmaps1.shape[1], heatmaps1.shape[1])) # images x heatmaps

    for img_id in tqdm(range(heatmaps1.shape[0])):#heatmaps1.shape):
        for h1_id in range(heatmaps1[img_id].shape[0]):
            _,hmap1 = preprocess_heatmap(heatmaps1,h1_id,img_id,THRESHOLD)
            for h2_id in range(heatmaps2[img_id].shape[0]):
                _,hmap2 = preprocess_heatmap(heatmaps2,h2_id,img_id,THRESHOLD)
                m_iou,_ = get_iou_purity(hmap1,hmap2)
                
                arr_ind_match[img_id,h1_id,h2_id]=m_iou
    return arr_ind_match


def preprocess_heatmap(heatmaps,hm_id,img_id,THRESHOLD):
    hmap = normalize(heatmaps[img_id,hm_id])
    hmap[hmap<THRESHOLD] = 0
    hmap[hmap>=THRESHOLD] = 0.5
    pred_mask_bool = (hmap>0.0).astype('bool')
    pred_mask_float = (hmap>0.0).astype(float)
    return pred_mask_float,pred_mask_bool

def get_iou_purity(gt_mask_bool,pred_mask_bool):
    overlap = gt_mask_bool*pred_mask_bool # Logical AND
    union = gt_mask_bool + pred_mask_bool # Logical OR
    if float(union.sum())>0.0:
        metric_iou = overlap.sum()/float(union.sum()) 
    else:
        metric_iou = 0.0
    if float(pred_mask_bool.sum())>0.0:
        metric_seg_purity = overlap.sum()/float(pred_mask_bool.sum())
    else:
        metric_seg_purity = 0.0   
    return metric_iou, metric_seg_purity

def calculate_purity_difference(images_np,heatmaps1, heatmaps2, THRESHOLD=0.3):
    arr_ind_match = calculate_heatmap_matches(np.array(heatmaps1, copy=True), 
                                              np.array(heatmaps2, copy=True))
    arr_purity_diffs = []
    arr_iou_diffs = []
    
    for img_id in tqdm(range(arr_ind_match.shape[0])):
        gt_mask_float,gt_mask_bool = preprocess_image(images_np,img_id)
        for hm1_id in range(arr_ind_match.shape[1]):
            hm2_id = np.argmax(arr_ind_match[img_id,hm1_id])

            pred_mask1_float,pred_mask1_bool = preprocess_heatmap(heatmaps1,hm1_id,img_id,THRESHOLD)
            pred_mask2_float,pred_mask2_bool = preprocess_heatmap(heatmaps2,hm2_id,img_id,THRESHOLD)

            metric_iou1, metric_seg_purity1 = get_iou_purity(gt_mask_bool,pred_mask1_bool)
            metric_iou2, metric_seg_purity2 = get_iou_purity(gt_mask_bool,pred_mask2_bool)
            arr_purity_diffs.append(metric_seg_purity2-metric_seg_purity1)
            arr_iou_diffs.append(metric_iou2-metric_iou1)
            '''
            print("metric_seg_purity 1 nmf", metric_seg_purity1)
            print("metric_seg_purity 2 snmf", metric_seg_purity2)
            print("metric_iou_b ", get_iou_purity(pred_mask1_bool,pred_mask2_bool))
            fig, ax = plt.subplots(1, 2,figsize=(7, 7*2))

            ax[0].imshow(1.0-images_np[img_id][:,:,0],cmap='gray')
            ax[0].imshow(pred_mask1_float,alpha=0.5)
            ax[0].axis('off')

            ax[1].imshow(1.0-images_np[img_id][:,:,0],cmap='gray')
            ax[1].imshow(pred_mask2_float,alpha=0.5)
            ax[1].axis('off')

            plt.show()
            '''
    return arr_purity_diffs,arr_iou_diffs



def preprocess_image(images_np,img_id):
    images_sel_draw = images_np
    if len(images_sel_draw.shape) < 4:
        images_sel_draw = np.expand_dims(images_sel_draw,0) #images_sel_draw.unsqueeze(0) #np.expand_dims(images_sel_draw,0)
    img_sel = images_sel_draw[img_id][:,:,0]
    img_sel =  1.0 - (img_sel>0.0).astype(float)
    gt_mask_float = img_sel
    gt_mask_bool = img_sel.astype('bool')
    return gt_mask_float,gt_mask_bool