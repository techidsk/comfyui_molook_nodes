import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


class MaskExpand:
    """
    提供多种mask扩张方法的节点。

    支持的扩张方法:
    - shapely: 使用Shapely库的buffer方法，提供精确的几何扩张，支持圆角/斜角/尖角
    - opencv: 使用OpenCV的膨胀操作，基于圆形结构元素
    - binary_dilation: 使用scipy的二值膨胀，简单快速的扩张
    - convex_hull: 先计算凸包，再进行可选的扩张

    参数说明:
    - masks: 输入的mask，支持批量处理
    - method: 扩张方法选择
    - distance: 扩张距离/半径
    - join_style: shapely方法的连接样式（round圆角/mitre尖角/bevel斜角）
    - invert: 是否反转结果

    返回:
    - masks: 扩张后的mask，保持输入维度不变
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "method": (
                    [
                        "shapely",
                        "dilate",
                        "binary_dilation",
                        "convex_hull",
                        "approx_poly",  # 新增多边形近似
                        "min_box",  # 新增最小外接矩形
                        "min_circle",  # 新增最小外接圆
                    ],
                ),
                "distance": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
                "join_style": (["round", "mitre", "bevel"],),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Molook_nodes/Mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    FUNCTION = "expand_mask"

    def _invert(self, mask: torch.Tensor):
        return 1 - mask

    def expand_mask(
        self,
        masks: torch.Tensor,
        method: str,
        distance: float,
        join_style: str = "round",
        invert: bool = False,
    ):
        """
        使用不同方法扩张mask
        """

        def _expand_single_mask(mask: torch.Tensor, dim: int = 3) -> torch.Tensor:
            # 将mask转换为numpy数组
            mask_np = np.clip(255.0 * mask.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )

            if method == "convex_hull":
                # 找到轮廓而不是非零点
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result_mask = np.zeros_like(mask_np)

                # 处理每个轮廓的凸包
                for contour in contours:
                    if len(contour) >= 3:
                        # 计算凸包
                        hull = cv2.convexHull(contour)
                        # 填充凸包区域
                        cv2.fillPoly(result_mask, [hull], 255)

                # 如果需要进一步扩张
                if distance > 0.1:
                    kernel_size = int(distance * 2)
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                    )
                    result_mask = cv2.dilate(result_mask, kernel, iterations=1)

            elif method == "shapely":
                # 找到轮廓
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # 转换轮廓为Shapely多边形
                polygons = []
                for contour in contours:
                    if len(contour) >= 3:  # 确保有足够的点形成多边形
                        poly = Polygon(contour.squeeze())
                        if poly.is_valid:  # 确保多边形有效
                            polygons.append(poly)

                if not polygons:
                    return mask

                # 合并多个多边形
                multi_poly = unary_union(polygons)

                # 扩张处理
                join_style_map = {"round": 1, "mitre": 2, "bevel": 3}
                expanded = multi_poly.buffer(
                    distance, join_style=join_style_map[join_style]
                )

                # 创建新的mask
                result_mask = np.zeros_like(mask_np)

                # 将扩张后的多边形转换回mask
                if isinstance(expanded, (Polygon, MultiPolygon)):
                    # 转换为整数坐标
                    if isinstance(expanded, Polygon):
                        expanded = [expanded]
                    else:
                        expanded = list(expanded.geoms)

                    for poly in expanded:
                        # 获取外部轮廓的整数坐标
                        exterior_coords = np.array(poly.exterior.coords).astype(
                            np.int32
                        )
                        cv2.fillPoly(result_mask, [exterior_coords], 255)

                        # 处理内部空洞
                        for interior in poly.interiors:
                            interior_coords = np.array(interior.coords).astype(np.int32)
                            cv2.fillPoly(result_mask, [interior_coords], 0)

            elif method == "dilate":
                kernel_size = int(distance * 2)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                result_mask = cv2.dilate(mask_np, kernel, iterations=1)

            elif method == "approx_poly":
                # 找到轮廓
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result_mask = np.zeros_like(mask_np)

                for contour in contours:
                    # 计算轮廓周长
                    perimeter = cv2.arcLength(contour, True)
                    # 修改epsilon的计算方式，使用更合理的比例
                    epsilon = (distance * 0.01) * perimeter  # 将系数缩小100倍
                    # 多边形近似
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    # 填充近似多边形
                    cv2.fillPoly(result_mask, [approx], 255)

            elif method == "min_box":
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result_mask = np.zeros_like(mask_np)

                for contour in contours:
                    # 计算最小外接矩形
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # 填充矩形
                    cv2.fillPoly(result_mask, [box], 255)

            elif method == "min_circle":
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                result_mask = np.zeros_like(mask_np)

                for contour in contours:
                    # 计算最小外接圆
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    # 绘制并填充圆
                    cv2.circle(result_mask, center, radius, 255, -1)

            else:  # binary_dilation
                result_mask = binary_dilation(mask_np, iterations=int(distance))
                result_mask = result_mask.astype(np.uint8) * 255

            # 转换回tensor
            if dim > 3:
                expanded_tensor = (
                    torch.from_numpy(result_mask.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .unsqueeze(1)
                )
            else:
                expanded_tensor = torch.from_numpy(
                    result_mask.astype(np.float32) / 255.0
                ).unsqueeze(0)

            return expanded_tensor

        if masks.ndim > 3:
            # 处理批量mask
            expanded_masks = []
            for mask in masks:
                expanded_mask = _expand_single_mask(mask, masks.ndim)
                expanded_masks.append(expanded_mask)
            expanded_tensor = torch.cat(expanded_masks, dim=0)
        else:
            # 处理单个mask
            expanded_tensor = _expand_single_mask(masks)

        if invert:
            expanded_tensor = self._invert(expanded_tensor)

        return (expanded_tensor,)


NODE_CLASS_MAPPINGS = {
    "MaskExpand(Molook)": MaskExpand
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskExpand(Molook)": "Mask Expand"
}
