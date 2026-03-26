from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8s.pt')
frame = np.zeros((320, 320, 3), dtype=np.uint8)
results = model.track(frame)
print('result type', type(results))
print('len', len(results))
if results:
    r = results[0]
    print('result attributes', dir(r))
    if hasattr(r, 'boxes'):
        print('boxes object', r.boxes)
        for b in r.boxes:
            print('box dir', dir(b))
            # try to get id attribute in multiple ways
            for attr in ['id', 'track_id', 'box_id']:
                if hasattr(b, attr):
                    print(f'found {attr}', getattr(b, attr))
            # inspect __dict__ maybe
            try:
                print('vars', vars(b))
            except:
                pass
