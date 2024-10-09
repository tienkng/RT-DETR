def calculate_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the width and height of the intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)

    # Compute the area of intersection rectangle
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the intersection over union
    iou = (
        interArea / float(box1Area + box2Area - interArea)
        if (box1Area + box2Area - interArea) > 0
        else 0
    )

    return iou


def remove_high_overlap_bboxes(bboxes, labels, scores, threshold=0.5):
    # bboxes is a tensor shape [N, (x1, y1, x2, y2)]
    # labels is a tensor shape [N]
    # scores is a tensor shape [N]
    to_remove = set()
    num_bboxes = len(bboxes)

    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            iou = calculate_iou(bboxes[i][:4], bboxes[j][:4])
            if iou > threshold:
                # Add the index of the box with class 2 to the removal set
                if labels[i] == 2 and labels[j] != 2:
                    to_remove.add(i)
                elif labels[j] == 2 and labels[i] != 2:
                    to_remove.add(j)
                else:
                    # If both boxes are not class 2, remove the one with the lower score
                    to_remove.add(i if scores[i] < scores[j] else j)

    return to_remove
