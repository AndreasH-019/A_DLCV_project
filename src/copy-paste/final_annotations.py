import pickle
import os

imgRoot = '../../data/coco_minitrain_25k/images_pruned/giraffes_elephants'
maskRoot = '../../data/coco_minitrain_25k/annotations'


with open(maskRoot+"/elephant_annotations.pickle", 'rb') as handle:
    eleAnnot = pickle.load(handle)

with open(maskRoot+"/giraffe_annotations.pickle", 'rb') as handle:
    girAnnot = pickle.load(handle)

finalEleAnnot = {}
finalGirAnnot = {}

for elephant in os.listdir("../../data/coco_minitrain_25k/annotations/elephant_masks_pruned"):
    finalEleAnnot[elephant] = eleAnnot[elephant]

for giraffe in os.listdir("../../data/coco_minitrain_25k/annotations/giraffe_masks_pruned"):
    finalGirAnnot[giraffe] = girAnnot[giraffe]

with open('../../data/coco_minitrain_25k/annotations/final_elephant_annotations.pickle', 'wb') as handle:
    pickle.dump(finalEleAnnot, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../../data/coco_minitrain_25k/annotations/final_giraffe_annotations.pickle', 'wb') as handle:
    pickle.dump(finalGirAnnot, handle, protocol=pickle.HIGHEST_PROTOCOL)

