"""ROS Detection3DArray serialization; topic fields and ID conventions unchanged."""
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

def detection_array(tracked, stamp, target_frame):
    det_array = Detection3DArray()
    det_array.header.stamp = stamp       # timestamp from RGB
    det_array.header.frame_id = target_frame    # b2/base_link

    for t in tracked:
        d3d = Detection3D()
        d3d.header = det_array.header

        if t['position'] is not None:
            d3d.bbox.center.position.x = float(t['position'][0])
            d3d.bbox.center.position.y = float(t['position'][1])
            d3d.bbox.center.position.z = float(t['position'][2])

        ext = t.get('extent', (0.0, 0.0, 0.0))
        d3d.bbox.size.x = float(ext[0])
        d3d.bbox.size.y = float(ext[1])
        d3d.bbox.size.z = float(ext[2])

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(t['class_name'])
        hyp.hypothesis.score = float(t['confidence'])
        d3d.results.append(hyp)

        d3d.id = str(t['track_id'])
        det_array.detections.append(d3d)

    return det_array
