"""快速排序（Quick Sort）的简单 Python 实现。

快速排序是一种基于「分治」思想的排序算法：
1. 从数组中选取一个基准元素（pivot）。
2. 将比基准小的元素放到左边，比基准大的元素放到右边（分区）。
3. 对左右两部分递归地执行上述过程。

平均时间复杂度 O(n log n)，最坏情况 O(n^2)。
"""

from typing import List


def quicksort(arr: List[int]) -> List[int]:
    """简洁版快速排序，返回一个新的有序列表（不修改原列表）。"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr: List[int], low: int = 0, high: int = None) -> List[int]:
    """原地快速排序，直接在传入的列表上排序并返回该列表。"""
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_index = _partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)
    return arr


def _partition(arr: List[int], low: int, high: int) -> int:
    """以 arr[high] 作为基准进行分区，返回基准最终所在的位置。"""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


if __name__ == "__main__":
    sample = [5, 2, 9, 1, 5, 6, 3, 8, 7, 0, 4]
    print("原始数据:", sample)
    print("快速排序:", quicksort(sample))
    print("原地排序:", quicksort_inplace(sample.copy()))
