import cv2
import numpy as np
import torch


class PixelSnappingNode:
    """
    使用SIFT特征匹配和仿射变换对齐两张图片，并拼接成全景图
    重叠区域只保留一次，输出包含两图完整内容的长方形
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),  # 参考图（图1）
                "target_image": ("IMAGE",),     # 待对齐图（图2）
                "max_features": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 20000,
                    "step": 100,
                    "display": "number"
                }),
                "match_ratio": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number"
                }),
                "ransac_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "invert_input_mask": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Inverted",
                    "label_off": "Normal"
                }),
                "mask_grow": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "mask_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "invert_output_mask": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Inverted",
                    "label_off": "Normal"
                }),
            },
            "optional": {
                "target_mask": ("MASK",),  # 图2的遮罩（可选）
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("stitched_image", "mask", "corrected_target")
    FUNCTION = "align_pixels"
    CATEGORY = "image/transform"
    
    def align_pixels(self, reference_image, target_image, max_features, match_ratio, ransac_threshold, 
                     invert_input_mask, mask_grow, mask_blur, invert_output_mask, target_mask=None):
        """
        使用SIFT特征匹配和仿射变换对齐图片，拼接成全景图
        
        Args:
            reference_image: 参考图（图1），ComfyUI格式 [B, H, W, C]
            target_image: 待对齐图（图2），ComfyUI格式 [B, H, W, C]
            max_features: SIFT特征点最大数量
            match_ratio: Lowe's ratio test阈值
            ransac_threshold: RANSAC算法的像素误差阈值
            invert_input_mask: 是否反转输入遮罩
            mask_grow: 遮罩扩张/收缩像素数（正数扩张，负数收缩）
            mask_blur: 遮罩模糊半径
            invert_output_mask: 是否反转输出遮罩
            target_mask: 图2的遮罩（可选），用于只覆盖特定区域
        
        Returns:
            拼接后的全景图（图1+图2，重叠区域只保留一次）
        """
        # 只处理batch中的第一张图
        ref_img = reference_image[0].cpu().numpy()  # [H, W, C]
        tgt_img = target_image[0].cpu().numpy()     # [H, W, C]
        
        # ComfyUI图像格式是 [0, 1] 范围的float32，转换为 [0, 255] 的uint8
        ref_img_uint8 = (ref_img * 255).astype(np.uint8)
        tgt_img_uint8 = (tgt_img * 255).astype(np.uint8)
        
        # 转换为灰度图用于特征检测
        if ref_img_uint8.shape[2] == 3:
            ref_gray = cv2.cvtColor(ref_img_uint8, cv2.COLOR_RGB2GRAY)
            tgt_gray = cv2.cvtColor(tgt_img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = ref_img_uint8[:, :, 0]
            tgt_gray = tgt_img_uint8[:, :, 0]
        
        # 创建SIFT检测器
        sift = cv2.SIFT_create(nfeatures=max_features)  # type: ignore
        
        # 检测特征点和计算描述符
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(tgt_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            print("警告: 特征点数量不足，返回原图")
            return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
        
        # 使用FLANN匹配器进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore
        
        # 保存原始图2用于重试
        tgt_img_uint8_original = tgt_img_uint8.copy()
        tgt_gray_original = tgt_gray.copy()
        
        # 外层大循环：最多尝试10次完整流程
        max_main_attempts = 10
        min_required_matches = 80
        good_matches = []  # 初始化
        
        affine_matrix = None
        ransac_mask = None
        src_pts = None
        dst_pts = None
        
        for main_attempt in range(max_main_attempts):
            print(f"\n{'='*60}")
            print(f"尝试 {main_attempt+1}/{max_main_attempts}")
            print(f"{'='*60}")
            
            # 每次大循环都从原始图2开始
            tgt_img_uint8 = tgt_img_uint8_original.copy()
            tgt_gray = tgt_gray_original.copy()
            
            # 检测特征点和计算描述符
            kp2, des2 = sift.detectAndCompute(tgt_gray, None)
            
            if des2 is None or len(kp2) < 3:
                if main_attempt < max_main_attempts - 1:
                    print("⚠️ 特征点不足，重试...")
                    continue
                else:
                    print("警告: 特征点数量不足，返回原图")
                    return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
            
            matches = flann.knnMatch(des2, des1, k=2)
            
            # Lowe's ratio test 筛选好的匹配点
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < match_ratio * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 3:
                if main_attempt < max_main_attempts - 1:
                    print(f"⚠️ 有效匹配点不足 ({len(good_matches)}个)，重试...")
                    continue
                else:
                    print(f"警告: 有效匹配点数量不足 ({len(good_matches)}个)，返回原图")
                    return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
            
            print(f"找到 {len(good_matches)} 个有效匹配点")
            
            # 提取匹配点的坐标
            src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore
            
            # 使用RANSAC估计仿射变换矩阵（完整仿射变换，支持非均匀缩放和剪切）
            affine_matrix, ransac_mask = cv2.estimateAffine2D(
                src_pts, 
                dst_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold
            )
            
            if affine_matrix is None:
                if main_attempt < max_main_attempts - 1:
                    print("⚠️ 无法计算仿射变换矩阵，重试...")
                    continue
                else:
                    print("警告: 无法计算仿射变换矩阵，返回原图")
                    return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
            
            inliers = np.sum(ransac_mask)
            print(f"RANSAC内点数量: {inliers}/{len(good_matches)}")
            
            # 【优化】基于重投影误差过滤低质量匹配点后重新计算
            inlier_mask = ransac_mask.ravel() == 1
            src_inliers = src_pts[inlier_mask]
            dst_inliers = dst_pts[inlier_mask]
            
            # 计算所有内点的重投影误差
            transformed = cv2.transform(src_inliers, affine_matrix)
            errors = np.linalg.norm(transformed - dst_inliers, axis=2).ravel()
            
            # 只保留误差最小的80%点
            threshold_80 = np.percentile(errors, 80)
            good_mask = errors <= threshold_80
            refined_count = np.sum(good_mask)
            
            print(f"🔧 重投影误差过滤: 保留{refined_count}/{inliers}个最优内点 (80%分位)")
            
            # 用最优点重新计算仿射矩阵
            if refined_count >= 3:  # 至少需要3个点
                src_pts_refined = src_inliers[good_mask]
                dst_pts_refined = dst_inliers[good_mask]
                affine_matrix_refined, _ = cv2.estimateAffine2D(
                    src_pts_refined, 
                    dst_pts_refined,
                    method=cv2.LMEDS  # 使用最小中值法，对离群点更鲁棒
                )
                
                if affine_matrix_refined is not None:
                    affine_matrix = affine_matrix_refined
                    print("✓ 已使用精化后的仿射矩阵")
            
            # 更新ransac_mask和src_pts/dst_pts以便后续使用
            if refined_count >= 3:
                # 重建完整的mask和点集（用于后续的包围框计算）
                temp_mask = np.zeros(len(good_matches), dtype=bool)
                inlier_indices = np.where(inlier_mask)[0]
                refined_indices = inlier_indices[good_mask]
                temp_mask[refined_indices] = True
                ransac_mask = temp_mask.reshape(-1, 1).astype(np.uint8)
            
            # 只使用RANSAC内点计算包围框
            inlier_mask = ransac_mask.ravel() == 1
            src_pts_inliers = src_pts[inlier_mask].reshape(-1, 2)  # 图2的内点
            dst_pts_inliers = dst_pts[inlier_mask].reshape(-1, 2)  # 图1的内点
            
            # 计算RANSAC内点的包围框（只用最可靠的匹配点）
            src_pts_2d = src_pts_inliers  # 图2中的RANSAC内点
            dst_pts_2d = dst_pts_inliers  # 图1中的RANSAC内点
            
            # 打印图片尺寸和匹配信息
            h1, w1 = ref_img_uint8.shape[:2]
            h2, w2 = tgt_img_uint8.shape[:2]
            print(f"图1尺寸: {w1}×{h1}")
            print(f"图2尺寸: {w2}×{h2}")
            
            # 图1匹配点的包围框
            dst_x_min, dst_y_min = np.min(dst_pts_2d, axis=0)
            dst_x_max, dst_y_max = np.max(dst_pts_2d, axis=0)
            dst_bbox_width = dst_x_max - dst_x_min
            dst_bbox_height = dst_y_max - dst_y_min
            
            # 图2匹配点的包围框
            src_x_min, src_y_min = np.min(src_pts_2d, axis=0)
            src_x_max, src_y_max = np.max(src_pts_2d, axis=0)
            src_bbox_width = src_x_max - src_x_min
            src_bbox_height = src_y_max - src_y_min
            
            print(f"图1包围框（基于{inliers}个内点）: {dst_bbox_width:.1f}×{dst_bbox_height:.1f}")
            print(f"图2包围框（基于{inliers}个内点）: {src_bbox_width:.1f}×{src_bbox_height:.1f}")
            
            # 计算缩放比例
            scale_a = dst_bbox_width / src_bbox_width if src_bbox_width > 0 else 1.0  # 宽度缩放比例
            scale_b = dst_bbox_height / src_bbox_height if src_bbox_height > 0 else 1.0  # 高度缩放比例
            
            print(f"数值 a (宽度比例): {scale_a:.3f}")
            print(f"数值 b (高度比例): {scale_b:.3f}")
            
            # 判断是否需要形变修正
            threshold = 0.006  # 0.6%的偏差阈值
            if abs(scale_a - scale_b) > threshold:
                print(f"="*60)
                print(f"⚠️ 检测到图2存在形变，a与b差异: {abs(scale_a - scale_b):.3f}")
                
                h2, w2 = tgt_img_uint8.shape[:2]
                print(f"当前图2尺寸: {w2}×{h2}")
                
                if scale_a > scale_b:
                    # a > b → 需要拉伸图2的宽度
                    # 修正后图2包围框的宽度 = 图1包围框宽度 × (图2包围框高度 / 图1包围框高度)
                    corrected_bbox_width = dst_bbox_width * (src_bbox_height / dst_bbox_height)
                    # 图2整图修正后的宽度 = 图2原图宽度 × (修正后图2包围框的宽度 / 图2包围框的宽度)
                    corrected_w = int(w2 * (corrected_bbox_width / src_bbox_width))
                    corrected_h = h2
                    print(f"修正后包围框宽度: {corrected_bbox_width:.1f} (原{src_bbox_width:.1f})")
                    print(f"🔧 横向拉伸修正 (a > b): {w2}×{h2} -> {corrected_w}×{corrected_h}")
                else:
                    # a < b → 需要拉伸图2的高度
                    # 修正后图2包围框的高度 = 图1包围框高度 × (图2包围框宽度 / 图1包围框宽度)
                    corrected_bbox_height = dst_bbox_height * (src_bbox_width / dst_bbox_width)
                    # 图2整图修正后的高度 = 图2原图高度 × (修正后图2包围框的高度 / 图2包围框的高度)
                    corrected_w = w2
                    corrected_h = int(h2 * (corrected_bbox_height / src_bbox_height))
                    print(f"修正后包围框高度: {corrected_bbox_height:.1f} (原{src_bbox_height:.1f})")
                    print(f"🔧 纵向拉伸修正 (a < b): {w2}×{h2} -> {corrected_w}×{corrected_h}")
                
                print(f"="*60)
                
                # 执行拉伸
                tgt_img_uint8 = cv2.resize(tgt_img_uint8, (corrected_w, corrected_h), interpolation=cv2.INTER_CUBIC)
                
                # 如果有输入遮罩，同步拉伸
                if target_mask is not None:
                    input_mask_original = target_mask[0].cpu().numpy()
                    if input_mask_original.shape[0] != h2 or input_mask_original.shape[1] != w2:
                        input_mask_original = cv2.resize(input_mask_original, (w2, h2), interpolation=cv2.INTER_LINEAR)
                    input_mask_corrected = cv2.resize(input_mask_original, (corrected_w, corrected_h), interpolation=cv2.INTER_LINEAR)
                
                # 重新转换为灰度图并重新匹配
                tgt_gray = cv2.cvtColor(tgt_img_uint8, cv2.COLOR_RGB2GRAY) if tgt_img_uint8.shape[2] == 3 else tgt_img_uint8[:, :, 0]
                
                print("🔍 使用修正后的图2重新进行SIFT匹配...")
                kp2, des2 = sift.detectAndCompute(tgt_gray, None)
                
                if des2 is None or len(kp2) < 3:
                    print("⚠️ 警告: 修正后特征点不足，回到大循环")
                    continue
                
                matches = flann.knnMatch(des2, des1, k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < match_ratio * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) < 3:
                    print(f"⚠️ 警告: 修正后有效匹配点不足 ({len(good_matches)}个)，回到大循环")
                    continue
                
                print(f"修正后找到 {len(good_matches)} 个有效匹配点")
                
                # 检查是否满足最小匹配点要求
                if len(good_matches) < min_required_matches:
                    print(f"⚠️ 修正后匹配点数量 {len(good_matches)} 低于 {min_required_matches}，回到大循环")
                    continue
                
                # 重新提取匹配点坐标（使用修正后的图2）
                src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore
                
                # 重新计算RANSAC仿射变换矩阵（完整仿射变换）
                affine_matrix, ransac_mask = cv2.estimateAffine2D(
                    src_pts, 
                    dst_pts, 
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_threshold
                )
                
                if affine_matrix is None:
                    print("⚠️ 警告: 修正后无法计算仿射变换，返回原图")
                    return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
                
                # 【优化】基于重投影误差过滤低质量匹配点后重新计算
                inliers_corrected = np.sum(ransac_mask)
                print(f"修正后RANSAC内点数量: {inliers_corrected}/{len(good_matches)}")
                
                inlier_mask_corrected = ransac_mask.ravel() == 1
                src_inliers_corrected = src_pts[inlier_mask_corrected]
                dst_inliers_corrected = dst_pts[inlier_mask_corrected]
                
                # 计算所有内点的重投影误差
                transformed_corrected = cv2.transform(src_inliers_corrected, affine_matrix)
                errors_corrected = np.linalg.norm(transformed_corrected - dst_inliers_corrected, axis=2).ravel()
                
                # 只保留误差最小的80%点
                threshold_80_corrected = np.percentile(errors_corrected, 80)
                good_mask_corrected = errors_corrected <= threshold_80_corrected
                refined_count_corrected = np.sum(good_mask_corrected)
                
                print(f"🔧 修正后重投影误差过滤: 保留{refined_count_corrected}/{inliers_corrected}个最优内点 (80%分位)")
                
                # 用最优点重新计算仿射矩阵
                if refined_count_corrected >= 3:
                    src_pts_refined_corrected = src_inliers_corrected[good_mask_corrected]
                    dst_pts_refined_corrected = dst_inliers_corrected[good_mask_corrected]
                    affine_matrix_refined_corrected, _ = cv2.estimateAffine2D(
                        src_pts_refined_corrected, 
                        dst_pts_refined_corrected,
                        method=cv2.LMEDS
                    )
                    
                    if affine_matrix_refined_corrected is not None:
                        affine_matrix = affine_matrix_refined_corrected
                        print("✓ 已使用修正后精化的仿射矩阵")
                        
                        # 更新ransac_mask
                        temp_mask_corrected = np.zeros(len(good_matches), dtype=bool)
                        inlier_indices_corrected = np.where(inlier_mask_corrected)[0]
                        refined_indices_corrected = inlier_indices_corrected[good_mask_corrected]
                        temp_mask_corrected[refined_indices_corrected] = True
                        ransac_mask = temp_mask_corrected.reshape(-1, 1).astype(np.uint8)
                
                # 验证修正效果（基于RANSAC内点）
                inlier_mask_new = ransac_mask.ravel() == 1
                src_pts_2d_new = src_pts[inlier_mask_new].reshape(-1, 2)
                dst_pts_2d_new = dst_pts[inlier_mask_new].reshape(-1, 2)
                
                src_bbox_w_new = np.max(src_pts_2d_new[:, 0]) - np.min(src_pts_2d_new[:, 0])  # type: ignore
                src_bbox_h_new = np.max(src_pts_2d_new[:, 1]) - np.min(src_pts_2d_new[:, 1])  # type: ignore
                dst_bbox_w_new = np.max(dst_pts_2d_new[:, 0]) - np.min(dst_pts_2d_new[:, 0])  # type: ignore
                dst_bbox_h_new = np.max(dst_pts_2d_new[:, 1]) - np.min(dst_pts_2d_new[:, 1])  # type: ignore
                
                scale_a_new = dst_bbox_w_new / src_bbox_w_new if src_bbox_w_new > 0 else 1.0
                scale_b_new = dst_bbox_h_new / src_bbox_h_new if src_bbox_h_new > 0 else 1.0
                
                print(f"修正后: a={scale_a_new:.3f}, b={scale_b_new:.3f}, 差异={abs(scale_a_new - scale_b_new):.3f}")
                print(f"✓ 改善效果: {abs(scale_a - scale_b):.3f} -> {abs(scale_a_new - scale_b_new):.3f}")
            else:
                print(f"✓ 图2无明显形变 (差异: {abs(scale_a - scale_b):.3f})")
            
            # 检查是否满足最小匹配点要求
            if len(good_matches) >= min_required_matches:
                print(f"✓ 匹配点数量满足要求 ({len(good_matches)}>={min_required_matches})")
                break
            else:
                if main_attempt < max_main_attempts - 1:
                    print(f"⚠️ 匹配点数量 {len(good_matches)} 低于 {min_required_matches}，回到循环开始重试...")
                    continue
                else:
                    print(f"⚠️ 达到最大尝试次数，匹配点数量 {len(good_matches)} 仍低于 {min_required_matches}，返回原图")
                    return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
        
        # 大循环结束后，最终检查是否满足条件
        if len(good_matches) < min_required_matches:
            print(f"⚠️ 最终检查: 匹配点数量 {len(good_matches)} 低于 {min_required_matches}，返回原图")
            return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
        
        # 确保变量已正确赋值
        if affine_matrix is None or ransac_mask is None or src_pts is None or dst_pts is None:
            print("警告: 处理异常，返回原图")
            return (target_image, torch.zeros(1, target_image.shape[1], target_image.shape[2]), target_image)
        
        # 保存修正后的图2用于输出
        corrected_target_img = tgt_img_uint8.astype(np.float32) / 255.0
        corrected_target_tensor = torch.from_numpy(corrected_target_img).unsqueeze(0)
        
        # 获取参考图的尺寸
        h1, w1 = ref_img_uint8.shape[:2]
        h2, w2 = tgt_img_uint8.shape[:2]
        
        # 使用图1作为画布尺寸（确保输出尺寸与图1一致）
        canvas_w = w1
        canvas_h = h1
        
        print(f"画布尺寸（使用图1尺寸）: {canvas_w}×{canvas_h}")
        
        # 创建画布，直接使用图1作为底图
        canvas = ref_img_uint8.copy()
        
        # 对图2应用仿射变换到画布上（直接变换到图1的坐标系）
        aligned_img_uint8 = cv2.warpAffine(
            tgt_img_uint8,
            affine_matrix,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 创建或处理遮罩
        if target_mask is not None:
            # 检查是否有修正后的遮罩（如果进行了形变修正）
            if 'input_mask_corrected' in locals() and input_mask_corrected is not None:  # type: ignore
                input_mask = input_mask_corrected  # type: ignore
                print("使用修正后的遮罩")
            else:
                # 使用输入的遮罩
                input_mask = target_mask[0].cpu().numpy()  # [H, W]
                
                # 确保遮罩尺寸与图2一致
                h2, w2 = tgt_img_uint8.shape[:2]
                if input_mask.shape[0] != h2 or input_mask.shape[1] != w2:
                    print(f"警告: 遮罩尺寸{input_mask.shape}与图2尺寸{(h2, w2)}不一致，进行缩放")
                    input_mask = cv2.resize(input_mask, (w2, h2), interpolation=cv2.INTER_LINEAR)
            
            # 反转输入遮罩（如果需要）
            if invert_input_mask:
                input_mask = 1.0 - input_mask
            
            # 将遮罩变换到画布上（使用原始affine_matrix）
            mask_2d = cv2.warpAffine(  # type: ignore
                input_mask,
                affine_matrix,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0  # type: ignore
            )
        else:
            # 不提供遮罩时，使用图2的完整区域（检测图2变换后的有效边界）
            # 创建图2原始尺寸的全1遮罩
            h2, w2 = tgt_img_uint8.shape[:2]
            full_mask = np.ones((h2, w2), dtype=np.float32)
            
            # 将完整遮罩变换到画布上（使用原始affine_matrix）
            mask_2d = cv2.warpAffine(  # type: ignore
                full_mask,
                affine_matrix,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0  # type: ignore
            )
        
        # 遮罩扩张/收缩
        if mask_grow != 0:
            if mask_grow > 0:
                # 扩张
                kernel_size = mask_grow * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_2d = cv2.dilate(mask_2d, kernel, iterations=1)
            else:
                # 收缩
                kernel_size = abs(mask_grow) * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_2d = cv2.erode(mask_2d, kernel, iterations=1)
        
        # 遮罩模糊
        if mask_blur > 0:
            kernel_size = int(mask_blur * 2) * 2 + 1  # 确保是奇数
            mask_2d = cv2.GaussianBlur(mask_2d, (kernel_size, kernel_size), mask_blur)
        
        # 确保遮罩值域在[0, 1]
        mask_2d = np.clip(mask_2d, 0.0, 1.0)
        
        mask_img = mask_2d[:, :, np.newaxis]  # [H, W, 1]
        
        # 将图2叠加到画布上（使用处理后的遮罩进行混合）
        final_img_uint8 = (canvas * (1 - mask_img) + aligned_img_uint8 * mask_img).astype(np.uint8)
        
        # 转换回ComfyUI格式
        final_img = final_img_uint8.astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_img).unsqueeze(0)
        
        # 输出遮罩（使用处理后的遮罩）
        output_mask = mask_2d.copy()
        
        # 反转输出遮罩（如果需要）
        if invert_output_mask:
            output_mask = 1.0 - output_mask
        
        mask_tensor = torch.from_numpy(output_mask).unsqueeze(0)  # [1, H, W]
        
        return (final_tensor, mask_tensor, corrected_target_tensor)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "PixelSnapping": PixelSnappingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelSnapping": "Pixel Snapping (SIFT)"
}