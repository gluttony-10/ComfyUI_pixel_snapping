import cv2
import numpy as np
import torch


class PowerfulMaskRestoreNode:
    """
    强力遮罩恢复节点
    将处理后的裁剪图像按照裁剪信息贴回到原图相应位置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),      # 原图
                "processed_crop": ("IMAGE",),      # 处理后的裁剪图（可能已缩放，如1024×1024）
                "original_mask": ("MASK",),        # 原图的遮罩（控制覆盖区域）
                "crop_info": ("CROP_INFO",),       # 裁剪信息（来自强力遮罩裁剪节点）
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_to_original"
    CATEGORY = "image/transform"
    
    def restore_to_original(self, original_image, processed_crop, original_mask, crop_info):
        """
        将处理后的裁剪图像恢复到原图位置
        
        Args:
            original_image: 原图 [B, H, W, C]
            processed_crop: 处理后的裁剪图 [B, H, W, C]（可能已缩放）
            original_mask: 原图的遮罩 [B, H, W]（控制覆盖区域）
            crop_info: 裁剪信息字典（包含original_size, crop_region, cropped_size, padding）
        
        Returns:
            恢复后的完整图像
        """
        # 提取裁剪信息
        orig_w, orig_h = crop_info["original_size"]
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_info["crop_region"]
        target_w, target_h = crop_info["cropped_size"]
        padding = crop_info.get("padding", {"top": 0, "bottom": 0, "left": 0, "right": 0})
        
        # 转换为numpy
        orig_img = original_image[0].cpu().numpy()  # [H, W, C]
        proc_crop = processed_crop[0].cpu().numpy()  # [H, W, C]
        orig_mask = original_mask[0].cpu().numpy()  # [H, W]
        
        # 验证原图尺寸
        if orig_img.shape[1] != orig_w or orig_img.shape[0] != orig_h:
            print(f"警告: 原图尺寸{orig_img.shape[1]}×{orig_img.shape[0]}与裁剪信息{orig_w}×{orig_h}不一致")
        
        # 裁剪区域的原始尺寸（在原图上的实际像素尺寸）
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min
        
        # 获取处理后裁剪图的实际尺寸
        proc_h, proc_w = proc_crop.shape[:2]
        
        print(f"原图尺寸: {orig_w}×{orig_h}")
        print(f"裁剪区域: ({crop_x_min}, {crop_y_min}) -> ({crop_x_max}, {crop_y_max})")
        print(f"目标尺寸: {target_w}×{target_h}")
        print(f"填充信息: {padding}")
        print(f"处理后图像尺寸: {proc_w}×{proc_h}")
        
        # 将处理后的图像缩放到目标尺寸
        if proc_w != target_w or proc_h != target_h:
            print(f"缩放处理图: {proc_w}×{proc_h} -> {target_w}×{target_h}")
            proc_crop_resized = cv2.resize(proc_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_crop_resized = proc_crop
        
        # 裁掉填充部分（只保留原图区域）
        valid_y_min = padding["top"]
        valid_y_max = target_h - padding["bottom"]
        valid_x_min = padding["left"]
        valid_x_max = target_w - padding["right"]
        
        print(f"有效区域: ({valid_x_min}, {valid_y_min}) -> ({valid_x_max}, {valid_y_max})")
        
        # 提取有效区域（去除填充）
        proc_crop_valid = proc_crop_resized[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
        
        # 将有效区域缩放回裁剪区域的原始尺寸
        print(f"缩放有效区域: {valid_x_max - valid_x_min}×{valid_y_max - valid_y_min} -> {crop_w}×{crop_h}")
        proc_crop_final = cv2.resize(proc_crop_valid, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建输出图像（原图的副本）
        result_img = orig_img.copy()
        
        # 提取原图遮罩中裁剪区域的部分
        # 确保遮罩尺寸与原图一致
        if orig_mask.shape[0] != orig_h or orig_mask.shape[1] != orig_w:
            print(f"警告: 遮罩尺寸{orig_mask.shape[1]}×{orig_mask.shape[0]}与原图不一致，进行缩放")
            orig_mask = cv2.resize(orig_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # 提取裁剪区域的遮罩
        crop_mask = orig_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # 将遮罩缩放到与处理后图像一致的尺寸
        if crop_mask.shape[0] != crop_h or crop_mask.shape[1] != crop_w:
            print(f"缩放遮罩: {crop_mask.shape[1]}×{crop_mask.shape[0]} -> {crop_w}×{crop_h}")
            crop_mask_resized = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        else:
            crop_mask_resized = crop_mask
        
        # 确保遮罩值域在[0, 1]
        crop_mask_resized = np.clip(crop_mask_resized, 0.0, 1.0)
        
        # 提取原图裁剪区域
        orig_crop_region = result_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # 使用遮罩混合
        mask_3d = crop_mask_resized[:, :, np.newaxis]
        blended = orig_crop_region * (1 - mask_3d) + proc_crop_final * mask_3d
        
        # 将混合结果贴回原图
        result_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = blended
        
        print(f"恢复完成，输出尺寸: {orig_w}×{orig_h}")
        
        # 转换回ComfyUI格式
        result_tensor = torch.from_numpy(result_img).unsqueeze(0)
        
        return (result_tensor,)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "PowerfulMaskRestore": PowerfulMaskRestoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerfulMaskRestore": "Powerful Mask Restore"
}
