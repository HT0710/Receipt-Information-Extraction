import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from rotation.CRAFT import model
from utils import download_weight


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = model.resize_aspect_ratio(image, 1536, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = model.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = model.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = model.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = model.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = model.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def model_setup(model, pretrained, cuda):
    if cuda:
        model.load_state_dict(copyStateDict(torch.load(pretrained)))
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
         model.load_state_dict(copyStateDict(torch.load(pretrained, map_location='cpu')))
    model.eval()
    return model
    
def setup(device):
    net = model.CRAFT()
    refine_net = model.RefineNet()
    cuda = True if (device == 'cuda' and torch.cuda.is_available()) else False
    
    net_model = download_weight('craft_mlt_25k.pth')
    refine_model = download_weight('craft_refiner_CTW1500.pth')

    net = model_setup(net, net_model, cuda)
    refine_net = model_setup(refine_net, refine_model, cuda)

    return net, refine_net, cuda
