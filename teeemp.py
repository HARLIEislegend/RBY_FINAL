import cv2
import os 
import datetime
import json
import numpy as np
with open(f'02_master/0_-03-27_1245__2026_30/all_results.json', 'r') as f:
    main_json = json.load(f)
    

color_dict = {
    "Red":    (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Blue":   (255, 0, 0)
}

flattened_list = []
print(list(main_json.keys()))
for frame_id,frame_data in main_json.items():
    for obj_id,obj_data in frame_data.items():
        x,y,w,h = obj_data['coords']
        c1 = x+(w)//2
        c2 = y+(h)//2
        flattened_list.append(
            {
            "frame": frame_id,
            "color": obj_data["color"],
            "coords": obj_data["coords"],
            "area": obj_data["area"],
            "center": (c1, c2)
            }
        )

# print(json.dumps(flattened_list,indent=4))
min_distance = 50 

cluster_coords = []

for elem in flattened_list:
    cen_x,cen_y = elem["center"]
    current_cluster = None

    for cluster in cluster_coords:
        print('one_cluster',cluster)
        mean_x = np.mean([singular_cluster['center'][0] for singular_cluster in cluster])
        mean_y = np.mean([singular_cluster['center'][1] for singular_cluster in cluster])
        print('check1')
        dist =  np.sqrt(np.abs(mean_x - cen_x)**2 + np.abs(mean_y - cen_y)**2)
        if dist<min_distance:
            current_cluster = cluster
            break
    if current_cluster is not None:
        current_cluster.append(elem)
    else:
        cluster_coords.append([elem])


    
final_data = {}
for i, group in enumerate(cluster_coords):
    avg_coords = np.mean([d["coords"] for d in group], axis=0).astype(int).tolist()
    avg_area   = int(np.mean([d["area"] for d in group]))
    avg_center = np.mean([d["center"] for d in group], axis=0).astype(int).tolist()
    
    colors = [d["color"] for d in group]
    most_freq_color = max(set(colors), key=colors.count)
    
    final_data[f"pole_{i}"] = {
        "color": most_freq_color,
        "avg_coords": avg_coords,
        "avg_center": avg_center,
        "avg_area": avg_area,
        "num_detections": len(group),
        "seen_in_frames": [d["frame"] for d in group]
    }

n1 = cv2.imread('02_master/0_-03-27_1746__2026_33/__4__/org_img.png',1)

folder_title = datetime.datetime.now().strftime("02_master/0_-%m-%d_%H%M__%Y_%S")
os.makedirs(folder_title,exist_ok=True)
print(final_data)
for pole,pole_data in final_data.items():
    
    x,y,w,h = pole_data['avg_coords']
    n1 = cv2.rectangle(n1,(x,y),(x+w,y+h),color_dict[pole_data['color']],1)
    

cv2.imwrite(f'{folder_title}/final_pred.png', n1)
with open(f'{folder_title}/final_results.json', 'w') as f:
    json.dump(final_data, f, indent=4)
