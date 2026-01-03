import cv2
import numpy as np
import torch


class PowerfulMaskCropNode:
    """
    强力遮罩裁剪节点
    根据遮罩区域裁剪图像，支持边缘保留和尺寸对齐到指定倍数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "mask": ("MASK",),    # 输入遮罩
                "top_padding": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "bottom_padding": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "left_padding": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "right_padding": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "enable_resize": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enabled",
                    "label_off": "Disabled"
                }),
                "target_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "display": "number"
                }),
                "target_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "display": "number"
                }),
                "padding_mode": (["black", "white", "none"],),
                "size_multiple": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "CROP_INFO")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_region_mask", "crop_info")
    FUNCTION = "crop_by_mask"
    CATEGORY = "image/transform"
    
    def crop_by_mask(self, image, mask, top_padding, bottom_padding, left_padding, right_padding,
                     enable_resize, target_width, target_height, padding_mode, size_multiple):
        """
        根据遮罩裁剪图像
        
        Args:
            image: 输入图像 [B, H, W, C]
            mask: 输入遮罩 [B, H, W]
            top_padding: 顶部保留像素
            bottom_padding: 底部保留像素
            left_padding: 左侧保留像素
            right_padding: 右侧保留像素
            enable_resize: 是否启用指定尺寸缩放
            target_width: 目标宽度（0表示自动）
            target_height: 目标高度（0表示自动）
            padding_mode: 填充模式（black/white/none）
            size_multiple: 尺寸对齐倍数（四舍五入到倍数）
        
        Returns:
            裁剪后的图像和遮罩
        """
        # 只处理batch中的第一张
        img = image[0].cpu().numpy()  # [H, W, C]
        msk = mask[0].cpu().numpy()   # [H, W]
        
        img_h, img_w = img.shape[:2]
        
        # 找到遮罩的边界框
        mask_binary = (msk > 0.5).astype(np.uint8)
        
        # 找到所有非零点
        coords = np.column_stack(np.where(mask_binary > 0))
        
        if len(coords) == 0:
            # 如果遮罩为空，返回原图
            print("警告: 遮罩为空，返回原图")
            return (image, mask)
        
        # 计算遮罩的边界
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        print(f"遮罩边界: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
        
        # 添加padding（保证不超出图像边界）
        crop_y_min = max(0, y_min - top_padding)
        crop_y_max = min(img_h, y_max + bottom_padding + 1)
        crop_x_min = max(0, x_min - left_padding)
        crop_x_max = min(img_w, x_max + right_padding + 1)
        
        # 保存初始裁剪区域坐标（用于恢复时定位）
        original_crop_x_min = crop_x_min
        original_crop_y_min = crop_y_min
        original_crop_x_max = crop_x_max
        original_crop_y_max = crop_y_max
        
        # 计算裁剪尺寸
        crop_h = crop_y_max - crop_y_min
        crop_w = crop_x_max - crop_x_min
        
        # 初始化填充信息（用于记录填充部分）
        padding_info = {
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0
        }
        
        print(f"添加padding后尺寸: {crop_w}×{crop_h}")
        
        # 执行裁剪
        cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        cropped_msk = msk[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        final_h, final_w = cropped_img.shape[:2]
        print(f"裁剪后尺寸: {final_w}×{final_h}")
        
        # 如果启用了指定尺寸缩放
        if enable_resize and (target_width > 0 or target_height > 0):
            # 确定目标宽高
            if target_width == 0:
                target_w = int(target_height * (final_w / final_h))
                target_h = target_height
            elif target_height == 0:
                target_w = target_width
                target_h = int(target_width * (final_h / final_w))
            else:
                target_w = target_width
                target_h = target_height
            
            if not enable_resize:
                # 未启用缩放，跳过
                pass
            else:
                # 拉伸模式：调整裁剪区域宽高比，然后缩放
                target_aspect = target_w / target_h
                current_aspect = final_w / final_h
                
                print(f"目标宽高比: {target_aspect:.3f}, 当前宽高比: {current_aspect:.3f}")
                
                if padding_mode == "none":
                    # 不填充模式：尝试扩展裁剪区域到目标宽高比，但不添加填充
                    # 如果原图边界不够，就保持原图能提供的最大尺寸
                    print("不填充模式: 扩展但不添加填充")
                    
                    if abs(target_aspect - current_aspect) > 0.01:  # 宽高比不一致
                        if target_aspect > current_aspect:
                            # 需要更宽，扩展宽度
                            new_crop_w = int(final_h * target_aspect)
                            w_expand = new_crop_w - final_w
                            print(f"需要扩展宽度 {w_expand} 像素")
                            
                            # 尝试从原图扩展，但不填充
                            left_expand = w_expand // 2
                            right_expand = w_expand - left_expand
                            
                            new_crop_x_min = max(0, crop_x_min - left_expand)
                            new_crop_x_max = min(img_w, crop_x_max + right_expand)
                            
                            # 更新裁剪区域坐标
                            crop_x_min = new_crop_x_min
                            crop_x_max = new_crop_x_max
                            
                            # 重新裁剪（不填充，使用原图实际能提供的尺寸）
                            cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            cropped_msk = msk[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            
                            actual_w = crop_x_max - crop_x_min
                            print(f"实际扩展后宽度: {actual_w} (目标{new_crop_w})")
                        else:
                            # 需要更高，扩展高度
                            new_crop_h = int(final_w / target_aspect)
                            h_expand = new_crop_h - final_h
                            print(f"需要扩展高度 {h_expand} 像素")
                            
                            # 尝试从原图扩展，但不填充
                            top_expand = h_expand // 2
                            bottom_expand = h_expand - top_expand
                            
                            new_crop_y_min = max(0, crop_y_min - top_expand)
                            new_crop_y_max = min(img_h, crop_y_max + bottom_expand)
                            
                            # 更新裁剪区域坐标
                            crop_y_min = new_crop_y_min
                            crop_y_max = new_crop_y_max
                            
                            # 重新裁剪（不填充，使用原图实际能提供的尺寸）
                            cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            cropped_msk = msk[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            
                            actual_h = crop_y_max - crop_y_min
                            print(f"实际扩展后高度: {actual_h} (目标{new_crop_h})")
                        
                        final_h, final_w = cropped_img.shape[:2]
                        print(f"扩展后尺寸: {final_w}×{final_h}")
                elif abs(target_aspect - current_aspect) > 0.01:  # 宽高比不一致
                    if target_aspect > current_aspect:
                        # 需要更宽，扩展宽度
                        new_crop_w = int(final_h * target_aspect)
                        w_expand = new_crop_w - final_w
                        print(f"需要扩展宽度 {w_expand} 像素")
                        
                        # 尝试从原图扩展
                        left_expand = w_expand // 2
                        right_expand = w_expand - left_expand
                        
                        new_crop_x_min = crop_x_min - left_expand
                        new_crop_x_max = crop_x_max + right_expand
                        
                        # 检查边界
                        left_padding_needed = 0
                        right_padding_needed = 0
                        
                        if new_crop_x_min < 0:
                            left_padding_needed = -new_crop_x_min
                            new_crop_x_min = 0
                        if new_crop_x_max > img_w:
                            right_padding_needed = new_crop_x_max - img_w
                            new_crop_x_max = img_w
                        
                        # 更新裁剪区域坐标
                        crop_x_min = new_crop_x_min
                        crop_x_max = new_crop_x_max
                        
                        # 重新裁剪
                        cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                        cropped_msk = msk[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                        
                        # 如果需要填充
                        if padding_mode != "none" and (left_padding_needed > 0 or right_padding_needed > 0):
                            # 确定填充值
                            if padding_mode == "white":
                                pad_value = 1.0
                            else:  # black
                                pad_value = 0.0
                            
                            cropped_img = np.pad(cropped_img, 
                                               ((0, 0), (left_padding_needed, right_padding_needed), (0, 0)),
                                               mode='constant', constant_values=pad_value)
                            cropped_msk = np.pad(cropped_msk,
                                               ((0, 0), (left_padding_needed, right_padding_needed)),
                                               mode='constant', constant_values=0)
                            print(f"填充: 左{left_padding_needed}px, 右{right_padding_needed}px ({padding_mode})")
                            
                            # 记录填充信息
                            padding_info["left"] += left_padding_needed
                            padding_info["right"] += right_padding_needed
                    else:
                        # 需要更高，扩展高度
                        new_crop_h = int(final_w / target_aspect)
                        h_expand = new_crop_h - final_h
                        print(f"需要扩展高度 {h_expand} 像素")
                        
                        # 尝试从原图扩展
                        top_expand = h_expand // 2
                        bottom_expand = h_expand - top_expand
                        
                        new_crop_y_min = crop_y_min - top_expand
                        new_crop_y_max = crop_y_max + bottom_expand
                        
                        # 检查边界
                        top_padding_needed = 0
                        bottom_padding_needed = 0
                        
                        if new_crop_y_min < 0:
                            top_padding_needed = -new_crop_y_min
                            new_crop_y_min = 0
                        if new_crop_y_max > img_h:
                            bottom_padding_needed = new_crop_y_max - img_h
                            new_crop_y_max = img_h
                        
                        # 更新裁剪区域坐标
                        crop_y_min = new_crop_y_min
                        crop_y_max = new_crop_y_max
                        
                        # 重新裁剪
                        cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                        cropped_msk = msk[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                        
                        # 如果需要填充
                        if padding_mode != "none" and (top_padding_needed > 0 or bottom_padding_needed > 0):
                            # 确定填充值
                            if padding_mode == "white":
                                pad_value = 1.0
                            else:  # black
                                pad_value = 0.0
                            
                            cropped_img = np.pad(cropped_img,
                                               ((top_padding_needed, bottom_padding_needed), (0, 0), (0, 0)),
                                               mode='constant', constant_values=pad_value)
                            cropped_msk = np.pad(cropped_msk,
                                               ((top_padding_needed, bottom_padding_needed), (0, 0)),
                                               mode='constant', constant_values=0)
                            print(f"填充: 上{top_padding_needed}px, 下{bottom_padding_needed}px ({padding_mode})")
                            
                            # 记录填充信息
                            padding_info["top"] += top_padding_needed
                            padding_info["bottom"] += bottom_padding_needed
                    
                    final_h, final_w = cropped_img.shape[:2]
                    print(f"调整宽高比后尺寸: {final_w}×{final_h}")
                
                # 记录缩放前的尺寸（用于计算缩放比例）
                before_scale_w = final_w
                before_scale_h = final_h
                
                # 执行缩放到目标尼寸
                if padding_mode == "none":
                    # 不填充模式：保持宽高比，按最小缩放比例
                    scale_w = target_w / final_w
                    scale_h = target_h / final_h
                    scale = min(scale_w, scale_h)
                    new_w = int(final_w * scale)
                    new_h = int(final_h * scale)
                    print(f"保持宽高比缩放到: {new_w}×{new_h}")
                else:
                    # 填充模式：拉伸到精确尺寸
                    new_w = target_w
                    new_h = target_h
                    print(f"缩放到: {new_w}×{new_h}")
                
                # 执行缩放
                cropped_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                cropped_msk = cv2.resize(cropped_msk, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                final_h, final_w = cropped_img.shape[:2]
                print(f"缩放后尺寸: {final_w}×{final_h}")
                
                # 计算缩放比例，更新填充信息
                scale_w_ratio = final_w / before_scale_w
                scale_h_ratio = final_h / before_scale_h
                padding_info["left"] = int(padding_info["left"] * scale_w_ratio)
                padding_info["right"] = int(padding_info["right"] * scale_w_ratio)
                padding_info["top"] = int(padding_info["top"] * scale_h_ratio)
                padding_info["bottom"] = int(padding_info["bottom"] * scale_h_ratio)
                print(f"缩放后填充: 左{padding_info['left']}, 右{padding_info['right']}, 上{padding_info['top']}, 下{padding_info['bottom']}")
        
        # 调整到指定倍数（最后执行，确保输出尺寸是倍数）
        if size_multiple > 1:
            final_w = round(final_w / size_multiple) * size_multiple
            final_h = round(final_h / size_multiple) * size_multiple
            
            # 确保至少是一个倍数
            if final_w == 0:
                final_w = size_multiple
            if final_h == 0:
                final_h = size_multiple
            
            print(f"对齐到{size_multiple}倍数: {final_w}×{final_h}")
            
            # 缩放到倍数尺寸
            cropped_img = cv2.resize(cropped_img, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
            cropped_msk = cv2.resize(cropped_msk, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
        
        print(f"最终输出尺寸: {final_w}×{final_h}")
        
        # 转换回ComfyUI格式
        cropped_img_tensor = torch.from_numpy(cropped_img).unsqueeze(0)
        cropped_msk_tensor = torch.from_numpy(cropped_msk).unsqueeze(0)
        
        # 创建裁剪区域遮罩（在原图尺寸上标记裁剪区域）
        crop_region_mask = np.zeros((img_h, img_w), dtype=np.float32)
        crop_region_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = 1.0
        crop_region_mask_tensor = torch.from_numpy(crop_region_mask).unsqueeze(0)
        
        # 保存裁剪信息用于后续恢复
        crop_info = {
            "original_size": (img_w, img_h),  # 原图尺寸 (宽, 高)
            "crop_region": (crop_x_min, crop_y_min, crop_x_max, crop_y_max),  # 最终裁剪区域（包括扩展后）
            "cropped_size": (final_w, final_h),  # 裁剪后的尺寸 (宽, 高)
            "padding": padding_info  # 填充信息（缩放后的填充像素数）
        }
        
        return (cropped_img_tensor, cropped_msk_tensor, crop_region_mask_tensor, crop_info)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "PowerfulMaskCrop": PowerfulMaskCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerfulMaskCrop": "Powerful Mask Crop"
}
