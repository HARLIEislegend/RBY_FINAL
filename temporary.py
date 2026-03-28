def inference_is_taken_here(frame,)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    frame = cv2.resize(frame, (640, 480))
    raw_image = frame
    raw_image_copy = raw_image.copy()
    raw_image_copy_diff = raw_image.copy()
    raw_image_final = raw_image.copy()

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth = depth_anything.infer_image(raw_image, input_sizee)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_img = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    depth_img_overlay = np.zeros_like(depth_img)

    # ── Optional enhancement ──────────────────────────────────────────────────
    org_img = boost_image(raw_image) if enhance_bool else raw_image

    # ── YOLO + SAM ────────────────────────────────────────────────────────────
    centers, yolo_annotated_img, orig_img = yolow_centers(depth_img, yolo_model, device)

    if sam_bool:
        masks, sam_overlay_img = sam_is_like_that(orig_img, centers, sam_predictor)
    else:
        masks = []
        sam_overlay_img = org_img

    # ── Mask processing ───────────────────────────────────────────────────────
    org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    combined_mask = np.zeros_like(org_img_rgb)
    combined_bool_mask = np.zeros((org_img_rgb.shape[0], org_img_rgb.shape[1]), dtype=bool)
    bounding_rect_list_with_color = []
    pole_masks = masks 


    folder_title = "00_master/1"
    os.makedirs(folder_title, exist_ok=True)
    if write_bool:
        folder_title = datetime.datetime.now().strftime("00_master/0_-%m-%d_%H%M__%Y_%S")
        os.makedirs(folder_title, exist_ok=True)

    for i, mask_array in enumerate(pole_masks):
        single_mask_visual = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        single_mask_visual[mask_array] = 255
        x, y, w, h = cv2.boundingRect(single_mask_visual)
        single_mask_bgr = cv2.cvtColor(single_mask_visual, cv2.COLOR_GRAY2BGR)

        if write_bool:
            cv2.imwrite(f'{folder_title}/thisismask{i}.png', single_mask_bgr)

        combined_bool_mask = np.logical_or(combined_bool_mask, mask_array)
        combined_mask = cv2.bitwise_or(combined_mask, single_mask_bgr)
        color_value = clasfiy_pole(org_img_rgb, mask_array, f"pole_repo_mask_{i}", folder_title, visualize=True)
        bounding_rect_list_with_color.append(([x, y, w, h], color_value))

    final_img = cv2.bitwise_and(combined_mask, org_img_rgb)
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

    for [x, y, w, h], color_value in bounding_rect_list_with_color:
        raw_image_copy_diff = cv2.rectangle(
            raw_image_copy_diff, (x, y), (x + w, y + h), color_dict[color_value], 4
        )

    # ── FPS overlay ───────────────────────────────────────────────────────────
    cv2.putText(raw_image_final, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ── Optional save ─────────────────────────────────────────────────────────
    if write_bool:
        cv2.imwrite(f'{folder_title}/depth_img.png', depth_img)
        cv2.imwrite(f'{folder_title}/final_img.png', final_img_bgr)
        cv2.imwrite(f'{folder_title}/enhanced_img.png', org_img)
        cv2.imwrite(f'{folder_title}/org_img.png', raw_image_copy)
        cv2.imwrite(f'{folder_title}/predicted_img.png', raw_image_copy_diff)
        cv2.imwrite(f'{folder_title}/depth.png', depth_img)
        clasfiy_pole(org_img_rgb, combined_bool_mask, "fin", folder_title, visualize=True)

    # ── Display ───────────────────────────────────────────────────────────────
    feed1= np.hstack((raw_image_final,depth_img))
    feed2 = np.hstack((final_img_bgr,raw_image_copy_diff))
    feed1 = np.vstack((feed1,feed2))