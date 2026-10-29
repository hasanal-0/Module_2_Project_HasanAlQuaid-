"""
AIPI 510: Module 2 Project: Statistics
- src/sorting_algorithms.py
Hasan Al-Quaid
=================================================
Merge sort, quicksort, and heapsort in pure Python.

Code adapted from GeeksforGeeks:
    Merge sort: https://www.geeksforgeeks.org/dsa/merge-sort/
    Quicksort:  https://www.geeksforgeeks.org/dsa/quick-sort-algorithm/
    Heapsort:   https://www.geeksforgeeks.org/dsa/heap-sort/

These follow the algorithms in Cormen et al. (2022)
"""

import random

# ---------------------------------------------------------------- Merge sort
def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid

    # Create temp arrays
    L = [0] * n1
    R = [0] * n2

    # Copy data to temp arrays L[] and R[]
    for i in range(n1):
        L[i] = arr[left + i]
    for j in range(n2):
        R[j] = arr[mid + 1 + j]
        
    i = 0  
    j = 0  
    k = left  

    # Merge the temp arrays back into arr[left..right]
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], if there are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort(arr, left, right):
    if left < right:
        mid = (left + right) // 2

        mergeSort(arr, left, mid)
        mergeSort(arr, mid + 1, right)
        merge(arr, left, mid, right)


# ----------------------------------------------------------------- Quicksort
def partition(arr, low, high):
    # choose the pivot
    pivot = arr[high]
    
    # index of smaller element and indicates the right position of pivot found so far
    i = low - 1
    
    # Traverse arr[low..high] and move all smaller elements to the left side
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            swap(arr, i, j)
    
    # move pivot after smaller elements and return its position
    swap(arr, i + 1, high)
    return i + 1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def quickSort(arr, low, high):
    if low < high:
        
        # pi is the partition return index of pivot
        pi = partition(arr, low, high)
        
        # recursion calls for smaller elements and greater or equals elements
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


# ------------------------------------------------------------------ Heapsort
def heapify(arr, n, i):
    # Initialize largest as root
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    # If left child is larger than root
    if l < n and arr[l] > arr[largest]:
        largest = l

    # If right child is larger than largest so far
    if r < n and arr[r] > arr[largest]:
        largest = r

    # If largest is not root, swap and heapify the affected sub-tree
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heapSort(arr):
    n = len(arr)

    # Build heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract an element from heap
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]

        # Call max heapify on the reduced heap
        heapify(arr, i, 0)


# ---------------------------------------------------- Wrappers for the study
def merge_sort(arr):
    mergeSort(arr, 0, len(arr) - 1)
    return arr

def quick_sort(arr):
    quickSort(arr, 0, len(arr) - 1)
    return arr

def heap_sort(arr):
    heapSort(arr)
    return arr

ALGORITHMS = {
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
}

def check_correctness(n_tests=200, seed=0):
    """
    Check every algorithm against Python's built-in sorted()
 
    Parameters:
        n_tests (int): Number of random test lists to generate
        seed (int): Random seed so the same test lists are generated every time
 
    Returns:
        int: The number of test lists checked, if every algorithm sorted them correctly
 
    Raises:
        AssertionError: If any algorithm's output doesn't match sorted()
    """
    random_num_gen = random.Random(seed)
 
    # 6 hand picked edge cases
    edge_cases = {
        "Empty list": [],
        "Single element": [1],
        "Two elements out of order": [2, 1],
        "All duplicates": [5, 5, 5],
        "Already sorted (0-99)": list(range(100)),
        "Sorted backwards (100-1)": list(range(100, 0, -1)),
    }
 
    # Random lists of random lengths
    random_tests = []
    for test_number in range(n_tests):
        length = random_num_gen.randint(0, 2000)
        random_list = []
        for position in range(length):
            random_list.append(random_num_gen.randint(-1000, 1000))
        random_tests.append(random_list)
 
    lengths = [len(test_list) for test_list in random_tests]
    print(f"Test lists: {len(edge_cases)} edge cases + {len(random_tests)} random lists ")
 
    all_passed = True
    for name, sort_function in ALGORITHMS.items():
        print(f"\n-------------------------- Start: {name} sort --------------------------")
 
        print("Edge cases:")
        edge_passed = 0
        for case_name, test_list in edge_cases.items():
            result = sort_function(test_list[:])
            passed = result == sorted(test_list)
            edge_passed += passed
            print(f"  {case_name:<28} length {len(test_list):>4}: {'PASS' if passed else 'FAIL'}")
 
        random_passed = 0
        for test_list in random_tests:
            result = sort_function(test_list[:])
            if result == sorted(test_list):
                random_passed += 1
 
        print(f"\nRandom lists passed: {random_passed}/{len(random_tests)}")
        print(f"Edge cases passed:   {edge_passed}/{len(edge_cases)}")
 
        algorithm_passed = (edge_passed == len(edge_cases)) and (random_passed == len(random_tests))
        print(f"All tests passed for {name} sort: {algorithm_passed}")
        print(f"-------------------------- End: {name} sort ----------------------------")
 
        if not algorithm_passed:
            all_passed = False
 
    if not all_passed:
        raise AssertionError("At least one algorithm failed a test")
    return len(edge_cases) + len(random_tests)
 
 
if __name__ == "__main__":
    total = check_correctness()
    print(f"\nAll {len(ALGORITHMS)} algorithms correctly sorted all {total} test lists.\n")

